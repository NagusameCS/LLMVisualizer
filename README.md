# LLMVisualizer

Web app that visualizes Ollama model metadata from a pasted Ollama Library link.

It does not download model weights. It only fetches and parses public model page metadata.

## Features

- Paste any `https://ollama.com/library/...` model link.
- Visual summary for selected variant, size, context window, and input type.
- Variant size map and detailed variant table.
- Lightweight local proxy to avoid browser CORS issues.

## Run Locally

1. Install dependencies:

```bash
npm install
```

2. Start the app:

```bash
npm start
```

3. Open:

```text
http://localhost:3000
```

## Example Inputs

- `https://ollama.com/library/llama3`
- `https://ollama.com/library/llama3:latest`
- `https://ollama.com/library/mistral`
- `https://ollama.com/library/qwen2.5:14b`

## Notes

- The app reads metadata from Ollama model and tags pages.
- If a model page format changes on Ollama, parser logic may need updates.