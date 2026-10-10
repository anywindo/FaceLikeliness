"""HTML/SVG visual gauge rendering for similarity score presentation."""

import math


def render_similarity_gauge(similarity_percent: float, category: str, cosine_sim: float) -> str:
    """Generates a modern, responsive HTML/SVG circular gauge for the similarity score.

    Args:
        similarity_percent: Normalized percentage in [0.0, 100.0].
        category: "High Similarity", "Moderate Similarity", or "Low Similarity".
        cosine_sim: Raw cosine similarity value in [-1.0, 1.0].

    Returns:
        str: Self-contained HTML string with scoped styles.
    """
    percent = max(0.0, min(100.0, float(similarity_percent)))

    # Circle math
    radius = 75
    circumference = 2 * math.pi * radius
    stroke_dashoffset = circumference - (percent / 100.0) * circumference

    # Category-specific color schemes
    if category == "High Similarity":
        gradient_start = "#10b981"  # Emerald 500
        gradient_end = "#06b6d4"    # Cyan 500
        glow_color = "rgba(16, 185, 129, 0.35)"
        badge_bg = "rgba(16, 185, 129, 0.15)"
        badge_border = "rgba(16, 185, 129, 0.4)"
        badge_text = "#34d399"
    elif category == "Moderate Similarity":
        gradient_start = "#f59e0b"  # Amber 500
        gradient_end = "#f97316"    # Orange 500
        glow_color = "rgba(245, 158, 11, 0.35)"
        badge_bg = "rgba(245, 158, 11, 0.15)"
        badge_border = "rgba(245, 158, 11, 0.4)"
        badge_text = "#fbbf24"
    else:
        gradient_start = "#f43f5e"  # Rose 500
        gradient_end = "#a855f7"    # Purple 500
        glow_color = "rgba(244, 63, 94, 0.35)"
        badge_bg = "rgba(244, 63, 94, 0.15)"
        badge_border = "rgba(244, 63, 94, 0.4)"
        badge_text = "#fb7185"

    html = f"""
    <div style="
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 24px 16px;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4), 0 0 20px {glow_color};
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        text-align: center;
        max-width: 320px;
        margin: 0 auto;
    ">
        <div style="position: relative; width: 190px; height: 190px;">
            <svg width="190" height="190" viewBox="0 0 190 190" style="transform: rotate(-90deg);">
                <defs>
                    <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stop-color="{gradient_start}" />
                        <stop offset="100%" stop-color="{gradient_end}" />
                    </linearGradient>
                    <filter id="gaugeShadow" x="-20%" y="-20%" width="140%" height="140%">
                        <feDropShadow dx="0" dy="0" stdDeviation="4" flood-color="{gradient_start}" flood-opacity="0.6"/>
                    </filter>
                </defs>
                <!-- Background track -->
                <circle
                    cx="95" cy="95" r="{radius}"
                    stroke="rgba(255, 255, 255, 0.08)"
                    stroke-width="12"
                    fill="transparent"
                />
                <!-- Animated Progress Arc -->
                <circle
                    cx="95" cy="95" r="{radius}"
                    stroke="url(#gaugeGradient)"
                    stroke-width="12"
                    stroke-linecap="round"
                    stroke-dasharray="{circumference}"
                    stroke-dashoffset="{stroke_dashoffset}"
                    fill="transparent"
                    filter="url(#gaugeShadow)"
                    style="transition: stroke-dashoffset 1s ease-in-out;"
                />
            </svg>
            <!-- Center Text -->
            <div style="
                position: absolute;
                top: 0; left: 0; right: 0; bottom: 0;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
            ">
                <span style="
                    font-size: 34px;
                    font-weight: 800;
                    letter-spacing: -0.5px;
                    color: #ffffff;
                    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
                    line-height: 1.1;
                ">
                    {percent:.1f}%
                </span>
                <span style="
                    font-size: 11px;
                    font-weight: 600;
                    letter-spacing: 1.5px;
                    text-transform: uppercase;
                    color: rgba(255, 255, 255, 0.5);
                    margin-top: 4px;
                ">
                    SIMILARITY
                </span>
            </div>
        </div>

        <!-- Qualitative Category Badge -->
        <div style="
            margin-top: 18px;
            padding: 6px 16px;
            border-radius: 9999px;
            background: {badge_bg};
            border: 1px solid {badge_border};
            color: {badge_text};
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0.5px;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        ">
            <span style="
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: {gradient_start};
                display: inline-block;
                box-shadow: 0 0 8px {gradient_start};
            "></span>
            {category}
        </div>

        <div style="
            margin-top: 10px;
            font-size: 12px;
            color: rgba(255, 255, 255, 0.45);
        ">
            Raw Cosine Sim: <strong style="color: rgba(255, 255, 255, 0.85);">{cosine_sim:+.4f}</strong>
        </div>
    </div>
    """
    return html
