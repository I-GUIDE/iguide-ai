// Which chrome this build wears. ONE source of truth, imported by both the header and the
// starter prompts so they can never disagree.
//
//   VITE_UI_VARIANT=platform   the ORIGINAL prototype page — I-GUIDE platform chrome,
//                              "Knowledge Elements", the generic starter prompts
//   (unset, default)           the rs-embed deployment from issue #20
//
// To switch back:
//   VITE_UI_VARIANT=platform npm run dev     (or `npm run build`)
//
// Written as a DIRECT string comparison on purpose. Vite inlines import.meta.env at build time,
// so `"platform" === "platform"` constant-folds and Rollup drops the branch that is not built —
// the unused variant is not shipped. An earlier version wrapped this in `?? 'rsembed'` and
// `.toLowerCase()`, which is not statically analysable: both variants then survived in every
// bundle, verified by grepping dist/ for each one's marker strings.
export const isPlatformVariant = import.meta.env.VITE_UI_VARIANT === 'platform';

// Which surface of the app is showing. A TEMPORARY demo tab sits beside the normal chat so
// the remote-sensing capabilities can be shown without a separate build or deployment: the
// ordinary page is untouched and remains the default, so removing the demo later is deleting
// a tab rather than unpicking a variant.
export type AppTab = 'chat' | 'rs';

export type UiVariant = 'platform' | 'rsembed';
export const UI_VARIANT: UiVariant = isPlatformVariant ? 'platform' : 'rsembed';
