import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { installFrontendLogging, reportError } from "./logging";
import "./styles.css";

installFrontendLogging();
void import("./App").then(({ default: App }) => createRoot(document.getElementById("root")!, {
  onUncaughtError: (error, info) => reportError("frontend.react.uncaught", { error, componentStack: info.componentStack }),
  onRecoverableError: (error, info) => reportError("frontend.react.recoverable", { error, componentStack: info.componentStack }),
}).render(
  <StrictMode>
    <ErrorBoundary><App /></ErrorBoundary>
  </StrictMode>,
)).catch((error: unknown) => {
  reportError("frontend.startup", error);
  const root = document.getElementById("root");
  if (root) root.textContent = "Image Studio could not start. See the application log for details.";
});
