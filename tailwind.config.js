module.exports = {
  content: [
    "./templates/**/*.html",
    "./*.html"
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        "on-background": "#191c1e",
        "secondary-fixed": "#f0dbff",
        "secondary": "#831ada",
        "tertiary-fixed-dim": "#ffb690",
        "surface-variant": "#e0e3e5",
        "on-tertiary-container": "#ffdecf",
        "on-secondary": "#ffffff",
        "on-tertiary": "#ffffff",
        "background": "#f7f9fb",
        "outline": "#7b7487",
        "on-secondary-fixed": "#2c0051",
        "on-surface": "#191c1e",
        "on-error-container": "#93000a",
        "secondary-fixed-dim": "#ddb8ff",
        "primary-fixed-dim": "#d2bbff",
        "on-error": "#ffffff",
        "error": "#ba1a1a",
        "error-container": "#ffdad6",
        "on-secondary-fixed-variant": "#6800b4",
        "on-surface-variant": "#4a4455",
        "inverse-surface": "#2d3133",
        "surface-tint": "#732ee4",
        "surface": "#f7f9fb",
        "on-secondary-container": "#fffbff",
        "surface-container-high": "#e6e8ea",
        "tertiary-container": "#aa4900",
        "surface-container-highest": "#e0e3e5",
        "surface-container-low": "#f2f4f6",
        "on-primary": "#ffffff",
        "primary": "#630ed4",
        "outline-variant": "#ccc3d8",
        "on-primary-fixed-variant": "#5a00c6",
        "on-primary-fixed": "#25005a",
        "on-tertiary-fixed": "#341100",
        "primary-container": "#7c3aed",
        "primary-fixed": "#eaddff",
        "surface-dim": "#d8dadc",
        "secondary-container": "#9e41f5",
        "surface-bright": "#f7f9fb",
        "surface-container-lowest": "#ffffff",
        "inverse-on-surface": "#eff1f3",
        "on-tertiary-fixed-variant": "#783200",
        "surface-container": "#eceef0",
        "inverse-primary": "#d2bbff",
        "tertiary-fixed": "#ffdbca",
        "on-primary-container": "#ede0ff",
        "tertiary": "#843700"
      },
      borderRadius: {
        "DEFAULT": "0.25rem",
        "lg": "0.5rem",
        "xl": "0.75rem",
        "full": "9999px"
      },
      spacing: {
        "gutter": "24px",
        "unit": "4px",
        "margin-desktop": "48px",
        "margin-mobile": "16px",
        "container-max": "1280px",
        "stack-md": "16px",
        "stack-lg": "32px",
        "stack-xl": "64px",
        "stack-sm": "8px"
      },
      fontFamily: {
        "display-lg": ["Inter"],
        "body-md": ["Inter"],
        "headline-md": ["Inter"],
        "body-lg": ["Inter"],
        "headline-sm": ["Inter"],
        "label-sm": ["Inter"],
        "display-lg-mobile": ["Inter"],
        "label-md": ["Inter"]
      },
      fontSize: {
        "display-lg": ["48px", { "lineHeight": "56px", "letterSpacing": "-0.02em", "fontWeight": "700" }],
        "body-md": ["16px", { "lineHeight": "24px", "fontWeight": "400" }],
        "headline-md": ["30px", { "lineHeight": "38px", "letterSpacing": "-0.01em", "fontWeight": "600" }],
        "body-lg": ["18px", { "lineHeight": "28px", "fontWeight": "400" }],
        "headline-sm": ["24px", { "lineHeight": "32px", "fontWeight": "600" }],
        "label-sm": ["12px", { "lineHeight": "16px", "letterSpacing": "0.02em", "fontWeight": "500" }],
        "display-lg-mobile": ["36px", { "lineHeight": "44px", "letterSpacing": "-0.02em", "fontWeight": "700" }],
        "label-md": ["14px", { "lineHeight": "20px", "letterSpacing": "0.01em", "fontWeight": "600" }]
      }
    }
  },
  plugins: []
}
