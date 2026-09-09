import { useState } from "react";
import { bridge } from "../bridge";
import { flushLogs, reportError } from "../logging";

export function LogTools() {
  const [error, setError] = useState("");
  const show = async () => {
    try {
      await flushLogs();
      await bridge.reveal(await bridge.getLogPath());
      setError("");
    } catch (reason) {
      reportError("frontend.logs.reveal", reason);
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  };
  return <div><button className="text-button" onClick={() => void show()}>Show application log</button>{error && <p className="error-banner" role="alert">{error}</p>}</div>;
}
