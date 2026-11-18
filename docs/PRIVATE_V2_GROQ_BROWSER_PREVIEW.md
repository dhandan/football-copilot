# Try Football Copilot V2 in your browser (private experimental preview)

**Status: experimental preview only, not the deployed production app.**
This runs V2, the real committed Premier League results/DuckDB tools, and the optional Groq GPT-OSS 20B planner with source-backed deterministic fallback. It shows exactly which tools were used and whether Groq rewrites were accepted, rejected, or bypassed.

## Preparation: keep the Free-plan ceiling

- GitHub Codespaces has a limited included allowance, **not unlimited free compute**. Review **Settings → Billing and licensing → Codespaces** before creating a Codespace. Do not enable paid overages. Codespaces availability depends on your GitHub account/organisation.
- Groq must remain on **Free**, with account **Zero Data Retention** enabled. The application checks your confirmation flags, not Groq's server-side settings.
- **A GitHub Actions secret does not automatically become a Codespaces secret.** In [Football Copilot's Codespaces secrets](https://github.com/dhandan/football-copilot/settings/secrets/codespaces), add secret **`GROQ_API_KEY`** with your existing Groq key; keep the key out of chat and repository files.

## Launch a private session

1. Open [the experimental Football Copilot branch](https://github.com/dhandan/football-copilot/tree/feature/issue9-cloud-readiness-audit).
2. Select **Code → Codespaces**, choose **New codespace** using the branch `feature/issue9-cloud-readiness-audit`. Check the branch selected on the create screen before proceeding. Choose the smallest available machine within your included free allowance. Do **not** use the `main` branch.
3. The experimental devcontainer sets up the existing cloud dependencies and starts the preview on **port 8501** when you attach. In the VS Code web IDE open the **PORTS** tab. **Confirm that port 8501 Visibility is PRIVATE** before opening its forwarded address. If the port isn't visible automatically, start it using `bash .devcontainer/start-preview.sh` in the Codespace terminal (no laptop Python required).
4. Open the **Forwarded Address** of port 8501 in the browser. The page title should be **Football Copilot V2** and the sidebar should show **Groq setup: Ready (user-attested)**. If not, check the secret was created under **Codespaces** and restart the Codespace.
5. Ask these questions, in this order (the last one is a conversational follow-up):
   - `Compare Liverpool and Arsenal over their last 10 matches`
   - `Who has been defensively better over those 10 matches?`
   - `Why is that?`
   - `Why was H07 rejected?`
6. Under each assistant response, expand **How this answer was produced**. The page displays the **read-only tools**, **sources**, **execution trace** and how many Groq requests were attempted. No API keys are shown.
7. After trying it, **Stop** (or Delete) the Codespace in the [Codespaces dashboard](https://github.com/codespaces) so it does not consume further allowance. Deleting removes the dev environment, not the GitHub repository.

## What to expect

- A real chat window, with session-scoped follow-ups and real football results.
- **Groq is used only when the experimental routing needs its model.** The recognised team-form examples use deterministic tools without contacting Groq; this is correct, not a failure.
- Some Groq-generated performance explanations still return HTTP 400 JSON validation or fail the grounded-answer gate; V2 then presents the verified deterministic response, with that fact visible in the trace. **Groq answer synthesis is currently NO-GO.**
- The historical/model-performance outputs have different freshness from live 2026/27 result ingestion. The private preview reads the **committed data snapshot**: it does not fetch the latest match results or update model weights.
- Only you should use this preview. It has **no application-level login**: privacy depends on Codespaces' authenticated **PRIVATE** forwarded port. **Never change port visibility to public** or share its URL. Do not enter sensitive personal data.
- One Codespace runs the application temporarily; it is not a permanently available free hosted chatbot.

## Engineering boundary

The preview is isolated in `app/v2_groq_preview.py`, `.devcontainer/devcontainer.json` and `.devcontainer/start-preview.sh` on draft PR #14 branch. It does not alter default Streamlit mode, forecasts, model weights, or production.
