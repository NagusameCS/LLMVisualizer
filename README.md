# LLMVisualizer

Static web app that visualizes Ollama model metadata from a pasted Ollama Library link.

It does not download model weights.

## Features

- Paste any `https://ollama.com/library/...` model link.
- Visual summary for selected variant, size, context window, and input type.
- Variant size map and detailed variant table.
- Compare two models side-by-side.
- Share/export results (PNG + JSON).

## Static Hosting (GitHub Pages)

This repository includes a Pages workflow at `.github/workflows/deploy-pages.yml`.

How it works:

1. Push to `main`.
2. GitHub Actions publishes the `public/` folder.
3. The app is served from site root `/` (no `/root` path needed).

No runtime install is required for users visiting the site.

## Local Preview (No Install)

You can open `public/index.html` directly in a browser.

## Example Inputs

- `https://ollama.com/library/llama3`
- `https://ollama.com/library/llama3:latest`
- `https://ollama.com/library/mistral`
- `https://ollama.com/library/qwen2.5:14b`

## Notes

- Metadata is fetched client-side via a CORS-friendly static proxy.
- If Ollama page markup changes, parsing logic may need updates.
- `Unchecked runtime.lastError: The message port closed before a response was received.` is usually from a browser extension and not from this app's code.