# Vertex Gemini Image Provider

Opt-in image generation backend for Hermes `image_generate`.

## Config

```yaml
image_gen:
  provider: vertex
  model: gemini-2.5-flash-image
  location: global
```

## Env

- `GOOGLE_GENAI_USE_VERTEXAI=true`
- `GOOGLE_CLOUD_PROJECT`
- `GOOGLE_CLOUD_LOCATION`
- optional `VERTEX_IMAGE_MODEL` / `GEMINI_IMAGE_MODEL` / `IMAGE_MODEL`

The provider uses Google Gen AI SDK in Vertex mode and returns the standard Hermes image provider response shape. Existing FAL/OpenAI/xAI/Krea providers are unchanged.
