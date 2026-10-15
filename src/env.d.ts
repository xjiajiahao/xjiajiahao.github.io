/// <reference types="astro/client" />

declare module "mermaid" {
  interface MermaidRunOptions {
    querySelector?: string;
  }

  interface MermaidConfig {
    startOnLoad?: boolean;
    securityLevel?: "strict" | "loose" | "antiscript" | "sandbox";
    theme?: string;
    fontFamily?: string;
    themeVariables?: Record<string, string>;
  }

  const mermaid: {
    initialize(config: MermaidConfig): void;
    run(options?: MermaidRunOptions): Promise<void>;
  };

  export default mermaid;
}
