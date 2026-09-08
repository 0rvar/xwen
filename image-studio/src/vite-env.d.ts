/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_BRIDGE_MODE?: "preview";
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
