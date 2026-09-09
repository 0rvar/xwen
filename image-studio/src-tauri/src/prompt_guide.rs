pub const ZIMAGE_PROMPT_GUIDE: &str = r#"
Z-Image Turbo prompting guide:
- Write natural prose, not tag soup. The Qwen3-4B text encoder reads sentences.
- Phrase constraints positively. Negative prompts and CFG adjustments do not help this distilled pipeline.
- Put the subject and framing first; the first roughly ten words matter most.
- A useful order is: framing, subject, hair or identifying details, posture and ground contact, view, facial action, body, skin detail, setting, lighting, and finish.
- Setting nouns and lighting conditions change the image most. Prefer concrete descriptions such as a wooden dock beside a calm forest lake, pale floorboards and white walls, or soft window light from the upper left.
- Describe visible actions instead of mood adjectives: laughing out loud with crinkled eyes is more useful than happy or confident.
- Physical contact makes poses clearer: hands flat against a wall, hands on knees, or barefoot on a named surface.
- Freckles, moles, tattoos, braids, and described film grain tend to show; generic claims like masterpiece, 8k, best quality, visible pores, candid, and documentary realism are weak or inert.
- Camera brands, lens names, nationality words, and most generic realism adjectives are usually weak. Describe the visual effect instead: gentle film grain, slightly overexposed on-camera flash, or soft overcast daylight.
- Keep prompts detailed prose, usually about 60–150 words, but do not pad them with empty quality tags. The pipeline truncates around 512 tokens.
- For portraits, 896x1152 is a useful starting size; 1024x1024 is safe. Use 8 steps by default; more steps usually buy little. Seeds vary less than prompt wording, so change prompts first and sweep a few seeds for selection.
- Full-body is not reliably enforced. State explicit ground contact and, when appropriate, that feet are visible and the camera is farther back.
- Avoid conflicting lighting instructions. Do not add a hair color to a monochrome prompt if selective color is unwanted.
"#;
