/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** 'platform' selects the ORIGINAL pre-#20 page; anything else (default) selects rs-embed. */
  readonly VITE_UI_VARIANT?: string;
  /** '1' builds the replay mode in (see src/replay.ts); dev builds always have it. */
  readonly VITE_REPLAY?: string;
}
interface ImportMeta {
  readonly env: ImportMetaEnv;
}
