// The official I-GUIDE hexagon mark.
//
// The asset is the brand logo from i-guide.io (`logo-color.png`, 492x240), cropped to the mark
// alone — the wordmark is dropped because the header already renders "I-GUIDE AI" as text
// beside it. Kept as a file rather than hand-authored SVG: the mark is a WOVEN pinwheel whose
// six chevrons are offset radially, which a rotated-path reconstruction cannot reproduce
// faithfully, and an approximate brand mark is worse than none.
//
// Native size is 208x240 (taller than wide), so it is sized by height and left to keep its own
// aspect ratio; `alt` is empty because the wrapping link carries the accessible name.
export function IGuideMark({ className }: { className?: string }) {
  return <img className={className} src="/iguide-mark.png" alt="" />;
}
