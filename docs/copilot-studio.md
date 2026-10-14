# Use the Insights Copilot from Microsoft Copilot Studio

The API's `/ask`, `/insights/{item_id}` and `/recommend/{user_id}` endpoints can be added to a
Copilot Studio agent as actions through a Power Platform custom connector. Category and
e-commerce teams then ask about products, reviews and recommendations in Teams or Microsoft 365
Copilot, and the answers come from this project's tools.

## 1. Deploy the API

Run the Docker image anywhere with HTTPS (for example Azure Container Apps or App Service):

```bash
docker build -t shelfsense .
docker run -p 8000:8000 \
  -e SHELFSENSE_API_TOKEN=<a long random key> \
  -e SHELFSENSE_PROVIDER=azure -e SHELFSENSE_MODEL=<your Azure OpenAI deployment> \
  -e AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com -e AZURE_OPENAI_API_KEY=<key> \
  shelfsense
```

With `SHELFSENSE_API_TOKEN` set, every request must send it in the `X-API-Key` header. Without an LLM
configured, `/ask` still works in the offline mock mode.

## 2. Create the custom connector

1. In [`copilot/insights-copilot.swagger.json`](../copilot/insights-copilot.swagger.json), set `host` to
   your API's host name.
2. In Power Apps or Power Automate, go to **Custom connectors**, choose **Import an OpenAPI
   file** and upload that file. (It is OpenAPI 2.0, which the custom connector wizard accepts.)
3. On the **Security** step choose **API Key**, header `X-API-Key`.
4. Create the connector, then use **Test** with a question such as "What do reviewers say about
   P0015?" and your key.

## 3. Add it to a Copilot Studio agent

1. Open your agent in Copilot Studio and add a tool (action) from the custom connector:
   **AskInsightsCopilot**, plus **GetProductInsights** and **RecommendProducts** for direct lookups.
2. Describe when to use it, for example: "Use for questions about product reviews, what
   shoppers say about taste or packaging, product recommendations and model monitoring."
3. Test in the test pane, then publish to Teams.

Menu names change between Power Platform releases; see Microsoft's documentation for
[custom connectors](https://learn.microsoft.com/en-us/connectors/custom-connectors/) and
[Copilot Studio](https://learn.microsoft.com/en-us/microsoft-copilot-studio/) if a step looks
different.

## Security notes

- Keep the key in the connector's connection, never in the OpenAPI file. Rotate it like any
  other secret.
- The API is read-only: it reads results and the warehouse, and never changes data.
- SQL from the agent is checked and runs read-only with a timeout, and shopper ids can only be
  counted, grouped or joined on, never returned (see the README's Guardrails for the limits).
