// Edge function: receives full CT, the cropped suspicious region, a heat-map
// crop, and DIP features. Asks Lovable AI vision for calibrated classification.
import "https://deno.land/x/xhr@0.1.0/mod.ts";
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type",
};

interface Features {
  area: number;
  areaRatio: number;
  meanIntensity: number;
  stdIntensity: number;
  circularity: number;
  solidity: number;
  edgeIrregularity: number;
  contrast: number;
  saliencyMean: number;
}

serve(async (req) => {
  if (req.method === "OPTIONS") return new Response(null, { headers: corsHeaders });

  try {
    const { imageBase64, regionBase64, heatBase64, features } = await req.json() as {
      imageBase64: string;
      regionBase64: string;
      heatBase64?: string;
      features: Features;
    };

    const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");
    if (!LOVABLE_API_KEY) throw new Error("LOVABLE_API_KEY missing");

    const systemPrompt = `You are an expert radiology demonstration assistant analyzing lung CT scans.
You are NOT a medical device. Output is for educational/research demonstration only.

You will see THREE images:
  1) The full CT slice.
  2) A cropped region of interest (ROI) detected by classical DIP.
  3) A "Grad-CAM-style" heat-map (red = high saliency).

You will also receive numeric features extracted from the ROI.

CALIBRATION RULES (follow strictly):
- First, judge whether the image is a lung CT slice. If not, set isLungCT=false,
  classification="indeterminate", tumorLikelihood<=0.15, and explain.
- Distinguish three cases for chest CTs:
   • NORMAL: clean lung parenchyma, no focal mass; ROI is likely vessel,
     fissure, or noise. tumorLikelihood typically 0.0-0.25.
   • BENIGN nodule: small (<2cm), well-circumscribed, smooth/round border,
     uniform density, may be calcified (very bright pinpoint). High circularity,
     high solidity, low edge irregularity. tumorLikelihood 0.25-0.55.
   • MALIGNANT-LOOKING: larger or irregular border, spiculated edges, low
     solidity, heterogeneous density, attached to pleura, surrounding distortion.
     tumorLikelihood 0.6-0.95.
- Be decisive: do NOT default to "benign" for everything.
- Use the heat-map: if the highlighted area sits on a clear focal opacity
  inside the lung field that contrasts with surrounding parenchyma, that
  supports a real finding.
- A single bright pinpoint with very high circularity and tiny area is more
  likely a vessel cross-section or calcification than a tumor.
- Edge irregularity > 0.5 + low solidity + medium-large area = malignant features.

Respond ONLY by calling the report_finding function.`;

    const userText = `ROI features (from DIP pipeline):
- area (px): ${features.area}
- area ratio of lung: ${(features.areaRatio * 100).toFixed(2)}%
- mean intensity (after CLAHE): ${features.meanIntensity.toFixed(1)} / 255
- std intensity: ${features.stdIntensity.toFixed(1)}
- circularity (1=perfect circle): ${features.circularity.toFixed(3)}
- solidity (1=convex): ${features.solidity.toFixed(3)}
- edge irregularity (0..1): ${features.edgeIrregularity.toFixed(3)}
- local contrast (0..1): ${features.contrast.toFixed(3)}
- mean saliency in ROI (0..1): ${features.saliencyMean.toFixed(3)}

Classify the case and explain.`;

    const userContent: unknown[] = [
      { type: "text", text: userText },
      { type: "image_url", image_url: { url: imageBase64 } },
      { type: "image_url", image_url: { url: regionBase64 } },
    ];
    if (heatBase64) {
      userContent.push({ type: "image_url", image_url: { url: heatBase64 } });
    }

    const body = {
      model: "google/gemini-2.5-flash",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userContent },
      ],
      tools: [
        {
          type: "function",
          function: {
            name: "report_finding",
            description: "Report calibrated tumor analysis finding for a lung CT.",
            parameters: {
              type: "object",
              properties: {
                isLungCT: {
                  type: "boolean",
                  description: "Whether the image plausibly looks like a lung CT slice.",
                },
                classification: {
                  type: "string",
                  enum: ["normal", "benign", "malignant", "indeterminate"],
                  description: "Overall case classification.",
                },
                tumorLikelihood: {
                  type: "number",
                  description: "0..1 probability that a real tumor is present.",
                },
                severity: {
                  type: "string",
                  enum: ["clear", "low-concern", "indeterminate", "suspicious", "highly-suspicious"],
                },
                rationale: {
                  type: "string",
                  description: "2-4 plain-language sentences citing visual + numeric evidence.",
                },
                recommendations: {
                  type: "array",
                  items: { type: "string" },
                  description: "1-3 short next-step suggestions.",
                },
              },
              required: ["isLungCT", "classification", "tumorLikelihood", "severity", "rationale", "recommendations"],
              additionalProperties: false,
            },
          },
        },
      ],
      tool_choice: { type: "function", function: { name: "report_finding" } },
    };

    const resp = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${LOVABLE_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    if (resp.status === 429) {
      return new Response(JSON.stringify({ error: "Rate limit. Try again shortly." }), {
        status: 429, headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
    if (resp.status === 402) {
      return new Response(
        JSON.stringify({ error: "AI credits exhausted. Add funds in Workspace > Usage." }),
        { status: 402, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }
    if (!resp.ok) {
      const t = await resp.text();
      console.error("AI gateway error:", resp.status, t);
      return new Response(JSON.stringify({ error: "AI gateway error" }), {
        status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    const data = await resp.json();
    const call = data.choices?.[0]?.message?.tool_calls?.[0];
    if (!call) {
      return new Response(JSON.stringify({ error: "No structured response" }), {
        status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
    const args = JSON.parse(call.function.arguments);

    return new Response(JSON.stringify(args), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (e) {
    console.error("analyze-scan error:", e);
    return new Response(JSON.stringify({ error: e instanceof Error ? e.message : "Unknown" }), {
      status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});
