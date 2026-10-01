/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** 'platform' selects the ORIGINAL pre-#20 page; anything else (default) selects rs-embed. */
  readonly VITE_UI_VARIANT?: string;
}
interface ImportMeta {
  readonly env: ImportMetaEnv;
}
