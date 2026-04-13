const express = require("express");
const cheerio = require("cheerio");

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.static("public"));

function normalizeModelInput(input) {
  if (!input || typeof input !== "string") {
    throw new Error("Missing model input.");
  }

  const trimmed = input.trim();
  let model = "";
  let tag = "";

  if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
    const url = new URL(trimmed);
    if (url.hostname !== "ollama.com") {
      throw new Error("Only ollama.com links are supported.");
    }

    const segments = url.pathname.split("/").filter(Boolean);
    if (segments[0] !== "library" || !segments[1]) {
      throw new Error("Use an Ollama library link like https://ollama.com/library/llama3.");
    }

    const modelTag = segments[1];
    const parts = modelTag.split(":");
    model = parts[0];
    tag = parts[1] || "";
  } else {
    const parts = trimmed.split(":");
    model = parts[0];
    tag = parts[1] || "";
  }

  if (!model) {
    throw new Error("Could not detect a model name from that input.");
  }

  return { model, tag };
}

function parseModelPage(html, modelName) {
  const $ = cheerio.load(html);
  const title = $("head title").text().trim() || modelName;
  const description =
    $("meta[name='description']").attr("content")?.trim() || "No description available.";

  const readmeTitle = $("#display h1").first().text().trim();
  const readmeParagraph = $("#display p").first().text().trim();

  const parameterHints = [];
  $("[x-test-size]").each((_, el) => {
    const size = $(el).text().trim();
    if (size) {
      parameterHints.push(size);
    }
  });

  return {
    title,
    description,
    readmeTitle,
    readmeParagraph,
    parameterHints: [...new Set(parameterHints)]
  };
}

function parseTagsPage(html, modelName) {
  const $ = cheerio.load(html);
  const variants = [];

  // Desktop rows are cleaner and easier to parse than the mobile duplicate layout.
  $("div.group.px-4.py-3").each((_, row) => {
    const desktop = $(row).find("div.hidden.md\\:flex").first();
    if (!desktop.length) {
      return;
    }

    const tagAnchor = desktop.find("div.grid a[href^='/library/']").first();
    const fullName = tagAnchor.text().trim();
    if (!fullName || !fullName.startsWith(`${modelName}:`)) {
      return;
    }

    const gridColumns = desktop.find("div.grid").first();
    const size = gridColumns.find("p.col-span-2").first().text().trim();
    const contextWindow = gridColumns.find("p.col-span-2").eq(1).text().trim();
    const inputType = gridColumns.find("div.col-span-2").first().text().trim();

    const metaLine = desktop.find("div.flex.text-neutral-500.text-xs").first().text();
    let digest = "";
    let updated = "";
    if (metaLine.includes("·")) {
      const bits = metaLine.split("·").map((v) => v.trim()).filter(Boolean);
      digest = bits[0] || "";
      updated = bits[1] || "";
    }

    variants.push({
      fullName,
      tag: fullName.split(":").slice(1).join(":"),
      size,
      contextWindow,
      inputType,
      digest,
      updated
    });
  });

  return variants;
}

function parseNumericSizeInGB(sizeText) {
  if (!sizeText) {
    return null;
  }

  const match = sizeText.match(/([\d.]+)\s*(GB|MB|TB)/i);
  if (!match) {
    return null;
  }

  const value = Number(match[1]);
  const unit = match[2].toUpperCase();
  if (!Number.isFinite(value)) {
    return null;
  }

  if (unit === "GB") {
    return value;
  }
  if (unit === "MB") {
    return value / 1024;
  }
  if (unit === "TB") {
    return value * 1024;
  }
  return null;
}

app.get("/api/model", async (req, res) => {
  try {
    const rawInput = req.query.url || req.query.model;
    const { model, tag } = normalizeModelInput(rawInput);

    const modelUrl = `https://ollama.com/library/${encodeURIComponent(model)}`;
    const tagsUrl = `${modelUrl}/tags`;

    const [modelResponse, tagsResponse] = await Promise.all([
      fetch(modelUrl),
      fetch(tagsUrl)
    ]);

    if (!modelResponse.ok) {
      return res.status(404).json({
        error: `Model page not found for '${model}'.`
      });
    }

    const modelHtml = await modelResponse.text();
    const tagsHtml = tagsResponse.ok ? await tagsResponse.text() : "";

    const modelInfo = parseModelPage(modelHtml, model);
    const variants = tagsHtml ? parseTagsPage(tagsHtml, model) : [];

    const variantsWithGb = variants.map((variant) => ({
      ...variant,
      sizeGb: parseNumericSizeInGB(variant.size)
    }));

    const selectedVariant = tag
      ? variantsWithGb.find((variant) => variant.tag === tag) || null
      : variantsWithGb[0] || null;

    return res.json({
      sourceInput: rawInput,
      model,
      tag,
      modelUrl,
      title: modelInfo.title,
      description: modelInfo.description,
      readmeTitle: modelInfo.readmeTitle,
      readmeParagraph: modelInfo.readmeParagraph,
      parameterHints: modelInfo.parameterHints,
      selectedVariant,
      variants: variantsWithGb,
      totalVariants: variantsWithGb.length,
      note: "Metadata only. This app does not download model weights."
    });
  } catch (error) {
    return res.status(400).json({ error: error.message || "Failed to parse model." });
  }
});

app.listen(PORT, () => {
  // eslint-disable-next-line no-console
  console.log(`LLM Visualizer running at http://localhost:${PORT}`);
});