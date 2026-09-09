import { Component, type ErrorInfo, type ReactNode } from "react";
import { reportError } from "../logging";

export class ErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  componentDidCatch(error: Error, info: ErrorInfo) {
    reportError("frontend.react", { error, componentStack: info.componentStack });
  }
  render() {
    if (this.state.failed) return <main className="fatal-screen"><h1>Image Studio encountered an error</h1><p>Check the application log for details.</p><button className="primary-button" onClick={() => location.reload()}>Reload Image Studio</button></main>;
    return this.props.children;
  }
}
