"""Premium Streamlit dashboard for retail shelf intelligence."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from html import escape
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pandas as pd
import streamlit as st

from analytics import analyze_detections, detect_gaps, evaluate_stock
from config import CONFIDENCE_THRESHOLD
from detector import ShelfDetector

RESULTS_STATE_KEY = "shelf_dashboard_results"
UPLOAD_WIDGET_KEY = "uploaded_shelf_image"


@dataclass(frozen=True)
class DashboardControls:
    """Interactive controls that shape the dashboard experience."""

    confidence_threshold: float
    show_bounding_boxes: bool
    show_analytics: bool


@st.cache_resource
def load_detector() -> ShelfDetector:
    """Load and cache the YOLO detector so the model is reused across reruns."""
    return ShelfDetector()


def inject_custom_css() -> None:
    """Apply a premium dashboard skin on top of the default Streamlit UI."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

        :root {
            --bg: #f3f7f6;
            --surface: rgba(255, 255, 255, 0.90);
            --surface-strong: #ffffff;
            --border: rgba(15, 23, 42, 0.08);
            --text: #102a2a;
            --muted: #5f7271;
            --accent: #0f766e;
            --accent-soft: #daf2ee;
            --highlight: #f59e0b;
            --success: #15803d;
            --danger: #b91c1c;
            --shadow: 0 22px 60px rgba(15, 23, 42, 0.08);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(15, 118, 110, 0.11), transparent 28%),
                radial-gradient(circle at top right, rgba(245, 158, 11, 0.10), transparent 24%),
                var(--bg);
            color: var(--text);
            font-family: 'Manrope', sans-serif;
        }

        .block-container {
            max-width: 1400px;
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
        }

        h1, h2, h3, h4, h5, h6 {
            color: var(--text);
            font-family: 'Space Grotesk', sans-serif;
            letter-spacing: -0.02em;
        }

        p, li, div, label, span {
            font-family: 'Manrope', sans-serif;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #092925 0%, #113a36 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        [data-testid="stSidebar"] * {
            color: #eef8f6;
        }

        [data-testid="stSidebar"] .stMarkdown p {
            color: rgba(238, 248, 246, 0.82);
        }

        [data-testid="stFileUploaderDropzone"] {
            background: rgba(255, 255, 255, 0.06);
            border: 1px dashed rgba(255, 255, 255, 0.25);
            border-radius: 18px;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 24px;
            box-shadow: var(--shadow);
            padding: 0.35rem 0.35rem 0.2rem 0.35rem;
            backdrop-filter: blur(10px);
            animation: fadeUp 0.45s ease both;
        }

        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, #ffffff 0%, #f8fbfa 100%);
            border: 1px solid rgba(15, 23, 42, 0.06);
            border-radius: 20px;
            padding: 1rem 1rem 0.9rem 1rem;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.95);
        }

        div[data-testid="stMetricLabel"] p {
            color: var(--muted);
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        div[data-testid="stMetricValue"] {
            color: var(--text);
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2rem;
            font-weight: 700;
        }

        [data-baseweb="tab-list"] {
            gap: 0.65rem;
            margin-bottom: 1rem;
        }

        [data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.76);
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 999px;
            color: var(--muted);
            font-weight: 600;
            padding: 0.5rem 1rem;
        }

        [data-baseweb="tab"][aria-selected="true"] {
            background: var(--accent);
            border-color: var(--accent);
            color: #ffffff;
        }

        [data-testid="stAlert"] {
            border-radius: 18px;
        }

        .hero-card {
            position: relative;
            overflow: hidden;
            padding: 1.45rem 1.55rem 1.35rem 1.55rem;
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 28px;
            background: linear-gradient(
                135deg,
                rgba(15, 118, 110, 0.16),
                rgba(255, 255, 255, 0.97) 42%,
                rgba(245, 158, 11, 0.15) 100%
            );
            box-shadow: var(--shadow);
            animation: fadeUp 0.45s ease both;
        }

        .hero-card::after {
            content: "";
            position: absolute;
            width: 12rem;
            height: 12rem;
            right: -3rem;
            bottom: -5rem;
            background: radial-gradient(circle, rgba(15, 118, 110, 0.14), transparent 72%);
        }

        .hero-eyebrow {
            color: var(--accent);
            font-size: 0.8rem;
            font-weight: 800;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            margin-bottom: 0.65rem;
        }

        .hero-title {
            margin: 0;
            font-size: clamp(2rem, 4vw, 3.4rem);
            line-height: 1;
        }

        .hero-subtitle {
            max-width: 42rem;
            margin: 0.7rem 0 0;
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.6;
        }

        .hero-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
            margin-top: 1.1rem;
        }

        .hero-badge {
            border-radius: 999px;
            border: 1px solid rgba(15, 118, 110, 0.14);
            background: rgba(255, 255, 255, 0.72);
            color: var(--text);
            font-size: 0.82rem;
            font-weight: 700;
            padding: 0.45rem 0.8rem;
        }

        .section-divider {
            height: 1px;
            margin: 1.15rem 0 0.75rem 0;
            background: linear-gradient(90deg, rgba(15, 118, 110, 0.5), rgba(15, 23, 42, 0.08));
        }

        .panel-heading {
            display: flex;
            align-items: center;
            gap: 0.9rem;
            margin-bottom: 1rem;
        }

        .panel-icon {
            align-items: center;
            background: linear-gradient(135deg, rgba(15, 118, 110, 0.14), rgba(245, 158, 11, 0.12));
            border-radius: 16px;
            display: flex;
            font-size: 1.15rem;
            height: 3rem;
            justify-content: center;
            width: 3rem;
        }

        .panel-kicker {
            color: var(--muted);
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            margin: 0 0 0.2rem 0;
            text-transform: uppercase;
        }

        .panel-title {
            margin: 0;
            font-size: 1.2rem;
        }

        .info-pills {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin: 0.35rem 0 1rem;
        }

        .info-pill {
            background: rgba(15, 118, 110, 0.08);
            border: 1px solid rgba(15, 118, 110, 0.1);
            border-radius: 999px;
            color: var(--text);
            font-size: 0.8rem;
            font-weight: 700;
            padding: 0.45rem 0.8rem;
        }

        .empty-state {
            position: relative;
            overflow: hidden;
            border-radius: 28px;
            border: 1px dashed rgba(15, 118, 110, 0.2);
            background: linear-gradient(135deg, rgba(15, 118, 110, 0.08), rgba(255, 255, 255, 0.86));
            padding: 2rem 1.6rem;
            min-height: 300px;
            box-shadow: var(--shadow);
            animation: fadeUp 0.45s ease both;
        }

        .empty-state::before {
            content: "";
            position: absolute;
            inset: auto auto -3.5rem -3.5rem;
            width: 10rem;
            height: 10rem;
            background: radial-gradient(circle, rgba(245, 158, 11, 0.16), transparent 72%);
        }

        .empty-kicker {
            color: var(--accent);
            font-size: 0.8rem;
            font-weight: 800;
            letter-spacing: 0.16em;
            text-transform: uppercase;
        }

        .empty-title {
            margin: 0.65rem 0 0;
            font-size: clamp(1.7rem, 3vw, 2.5rem);
            line-height: 1.1;
        }

        .empty-copy {
            color: var(--muted);
            margin: 0.85rem 0 0;
            max-width: 34rem;
            line-height: 1.7;
        }

        .summary-list {
            display: grid;
            gap: 0.75rem;
            margin-top: 1rem;
        }

        .summary-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.85rem 0.95rem;
            border-radius: 18px;
            background: rgba(15, 118, 110, 0.05);
            border: 1px solid rgba(15, 118, 110, 0.08);
        }

        .summary-label {
            color: var(--muted);
            font-size: 0.85rem;
            font-weight: 600;
        }

        .summary-value {
            color: var(--text);
            font-weight: 800;
        }

        .gap-emphasis {
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(15, 118, 110, 0.09), rgba(255, 255, 255, 0.98));
            border: 1px solid rgba(15, 23, 42, 0.08);
            padding: 1.3rem 1.2rem;
            margin-bottom: 1rem;
        }

        .gap-label {
            color: var(--muted);
            display: block;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }

        .gap-value {
            color: var(--text);
            display: block;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 3rem;
            font-weight: 700;
            line-height: 1.05;
            margin: 0.4rem 0 0.35rem;
        }

        .gap-copy {
            color: var(--muted);
            margin: 0;
            line-height: 1.6;
        }

        .caption-note {
            color: var(--muted);
            font-size: 0.84rem;
            margin-top: 0.5rem;
        }

        @keyframes fadeUp {
            from {
                opacity: 0;
                transform: translateY(14px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Render the hero header for the dashboard."""
    st.markdown(
        """
        <section class="hero-card">
            <div class="hero-eyebrow">AI Shelf Monitoring Dashboard</div>
            <h1 class="hero-title">🛒 Retail Shelf Intelligence System</h1>
            <p class="hero-subtitle">AI-powered shelf monitoring and analytics</p>
            <div class="hero-badges">
                <span class="hero-badge">YOLO Detection</span>
                <span class="hero-badge">Shelf Gap Visibility</span>
                <span class="hero-badge">Low-Stock Monitoring</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


def render_panel_heading(icon: str, title: str, kicker: str) -> None:
    """Render a reusable heading for dashboard cards."""
    st.markdown(
        f"""
        <div class="panel-heading">
            <div class="panel-icon">{icon}</div>
            <div>
                <p class="panel-kicker">{kicker}</p>
                <h3 class="panel-title">{title}</h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[Any | None, DashboardControls]:
    """Render the sidebar upload and dashboard controls."""
    with st.sidebar:
        st.markdown("## Control Center")
        st.caption("Tune the analysis workflow before reviewing the shelf output.")

        # Improvement 3: let users clear analysis state and upload a new image quickly.
        if st.button(
            "🔄 Reset Analysis",
            use_container_width=True,
            help="Clear the current upload and analysis results.",
        ):
            reset_dashboard_state()

        uploaded_file = st.file_uploader(
            "Upload shelf image",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG",
            key=UPLOAD_WIDGET_KEY,
        )

        confidence_threshold = st.slider(
            "Confidence threshold",
            min_value=0.1,
            max_value=1.0,
            value=float(CONFIDENCE_THRESHOLD),
            step=0.05,
            help="Lower values surface more detections but can introduce more noise.",
        )
        show_bounding_boxes = st.toggle("Show bounding boxes", value=True)
        show_analytics = st.toggle("Show analytics", value=True)

        st.markdown("---")
        st.markdown("### About Project")
        st.markdown(
            """
            This dashboard combines YOLO-powered object detection with shelf analytics to help retail teams:

            - Monitor product presence
            - Surface low-stock risks
            - Detect visible shelf gaps
            - Review category-level inventory distribution
            """
        )
        st.caption("Designed for modern retail ops, merchandising, and store intelligence workflows.")

    return uploaded_file, DashboardControls(
        confidence_threshold=confidence_threshold,
        show_bounding_boxes=show_bounding_boxes,
        show_analytics=show_analytics,
    )


def save_uploaded_image(file_bytes: bytes, filename: str) -> Path:
    """Persist uploaded image bytes to a temporary file for YOLO processing."""
    suffix = Path(filename).suffix or ".jpg"

    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(file_bytes)
        return Path(temp_file.name)


def reset_dashboard_state() -> None:
    """Clear session state so the app returns to a fresh upload state."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]

    st.rerun()


def get_upload_signature(file_bytes: bytes, confidence_threshold: float) -> str:
    """Create a cache key that changes only when analysis inputs change."""
    digest = hashlib.sha256(file_bytes).hexdigest()
    return f"{digest}:{confidence_threshold:.2f}"


def build_category_dataframe(category_counts: dict[str, int]) -> pd.DataFrame:
    """Transform category counts into a chart-friendly dataframe."""
    if not category_counts:
        return pd.DataFrame(columns=["Category", "Count", "Share"])

    category_df = pd.DataFrame(
        [{"Category": category_name.title(), "Count": count} for category_name, count in category_counts.items()]
    ).sort_values("Count", ascending=False)
    total_items = int(category_df["Count"].sum()) or 1
    category_df["Share"] = (category_df["Count"] / total_items * 100).round(1)
    return category_df


def get_primary_category(category_counts: dict[str, int]) -> str:
    """Return the highest-volume detected category."""
    if not category_counts:
        return "No items detected"

    category_name, count = max(category_counts.items(), key=lambda item: item[1])
    return f"{category_name.title()} ({count})"


def process_uploaded_image(
    file_bytes: bytes,
    filename: str,
    confidence_threshold: float,
) -> dict[str, Any]:
    """Run the detection pipeline and package dashboard-ready results."""
    temp_image_path = save_uploaded_image(file_bytes, filename)

    try:
        detector = load_detector()
        detections = detector.detect(
            str(temp_image_path),
            confidence_threshold=confidence_threshold,
        )
        analysis = analyze_detections(detections)
        alerts = evaluate_stock(analysis)
        gap_count = detect_gaps(detections)
        annotated_image = detector.get_annotated_frame()

        return {
            "file_name": filename,
            "original_image": file_bytes,
            "annotated_image": annotated_image,
            "analysis": analysis,
            "alerts": alerts,
            "gap_count": gap_count,
            "category_df": build_category_dataframe(analysis["category_counts"]),
            "primary_category": get_primary_category(analysis["category_counts"]),
        }
    finally:
        temp_image_path.unlink(missing_ok=True)


def get_or_process_results(
    file_bytes: bytes,
    filename: str,
    controls: DashboardControls,
) -> tuple[dict[str, Any], bool]:
    """Reuse prior results when possible and process only when inputs change."""
    signature = get_upload_signature(file_bytes, controls.confidence_threshold)
    cached_state = st.session_state.get(RESULTS_STATE_KEY)

    if cached_state and cached_state["signature"] == signature:
        return cached_state["payload"], False

    # Improvement 2: show explicit processing feedback while the pipeline runs.
    with st.spinner("🔍 Analyzing shelf..."):
        payload = process_uploaded_image(
            file_bytes=file_bytes,
            filename=filename,
            confidence_threshold=controls.confidence_threshold,
        )

    st.session_state[RESULTS_STATE_KEY] = {
        "signature": signature,
        "payload": payload,
    }
    return payload, True


def render_empty_state() -> None:
    """Render a polished landing state before any image is uploaded."""
    left_col, right_col = st.columns([1.35, 0.85], gap="large")

    with left_col:
        st.markdown(
            """
            <section class="empty-state">
                <div class="empty-kicker">Ready for analysis</div>
                <h2 class="empty-title">Upload a shelf image to unlock operational intelligence.</h2>
                <p class="empty-copy">
                    Review annotated detections, understand category distribution, surface low-stock
                    risks, and identify visible shelf gaps from one clean AI dashboard.
                </p>
            </section>
            """,
            unsafe_allow_html=True,
        )

    with right_col:
        with st.container(border=True):
            render_panel_heading("✨", "What you will see", "Dashboard preview")
            st.markdown(
                """
                - Annotated shelf imagery with clean visual focus
                - KPI cards for total items, categories, and gaps
                - Category breakdown charts for shelf mix visibility
                - Low-stock alerts with operational severity cues
                - Gap analysis for on-shelf availability checks
                """
            )
            st.markdown(
                '<p class="caption-note">Start from the sidebar to load an image and generate insights.</p>',
                unsafe_allow_html=True,
            )


def render_analysis_meta(results: dict[str, Any], controls: DashboardControls) -> None:
    """Render compact status pills above the main dashboard content."""
    escaped_filename = escape(results["file_name"])
    escaped_primary_category = escape(results["primary_category"])

    st.markdown(
        f"""
        <div class="info-pills">
            <span class="info-pill">📁 {escaped_filename}</span>
            <span class="info-pill">🎯 Confidence {controls.confidence_threshold:.2f}</span>
            <span class="info-pill">🚨 {len(results["alerts"])} alerts</span>
            <span class="info-pill">🏷️ Primary: {escaped_primary_category}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_image_panel(results: dict[str, Any], controls: DashboardControls) -> None:
    """Render the shelf image card in the left column."""
    show_annotated_image = (
        controls.show_bounding_boxes and results["annotated_image"] is not None
    )
    display_image = results["annotated_image"] if show_annotated_image else results["original_image"]
    image_caption = "Annotated shelf image" if show_annotated_image else "Original shelf image"

    with st.container(border=True):
        render_panel_heading("🖼️", "Shelf View", "Visual intelligence")
        if show_annotated_image:
            st.image(
                display_image,
                caption=image_caption,
                channels="BGR",
                use_container_width=True,
            )
        else:
            st.image(
                display_image,
                caption=image_caption,
                use_container_width=True,
            )
        st.markdown(
            f"""
            <p class="caption-note">
                Bounding boxes are {"enabled" if controls.show_bounding_boxes else "hidden"}.
                Use the sidebar to switch between the annotated output and the raw shelf image.
            </p>
            """,
            unsafe_allow_html=True,
        )


def render_summary_panel(results: dict[str, Any], controls: DashboardControls) -> None:
    """Render KPI metrics and a short executive summary in the right column."""
    analysis = results["analysis"]
    alerts = results["alerts"]
    gap_count = results["gap_count"]
    escaped_primary_category = escape(results["primary_category"])

    with st.container(border=True):
        render_panel_heading("📊", "Executive Snapshot", "Shelf health at a glance")
        metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
        metric_col_1.metric("Total Items", analysis["total_items"])
        # Improvement 4: business-friendly KPI labels.
        metric_col_2.metric("Unique Categories", len(analysis["category_counts"]))
        metric_col_3.metric("Shelf Gaps", gap_count)
        st.caption(
            "Unique Categories counts distinct detected product classes. "
            "Shelf Gaps highlights visible horizontal spacing between adjacent items."
        )

        st.markdown(
            f"""
            <div class="summary-list">
            <div class="summary-item">
                <span class="summary-label">Primary category</span>
                <span class="summary-value">{escaped_primary_category}</span>
            </div>
            <div class="summary-item">
                <span class="summary-label">Low-stock alerts</span>
                <span class="summary-value">{len(alerts)}</span>
            </div>
            <div class="summary-item">
                <span class="summary-label">Overlay mode</span>
                <span class="summary-value">{"On" if controls.show_bounding_boxes else "Off"}</span>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_category_breakdown_tab(results: dict[str, Any]) -> None:
    """Render category analytics using a chart and a sortable table."""
    category_df: pd.DataFrame = results["category_df"]

    if category_df.empty:
        st.info("No categories were detected in the uploaded shelf image.")
        return

    chart_col, table_col = st.columns([1.2, 1], gap="large")
    with chart_col:
        st.bar_chart(
            category_df.set_index("Category")["Count"],
            use_container_width=True,
        )
    with table_col:
        st.dataframe(
            category_df,
            hide_index=True,
            use_container_width=True,
        )


def render_alerts_tab(results: dict[str, Any]) -> None:
    """Render low-stock status using high-visibility alert components."""
    alerts: list[str] = results["alerts"]

    if not alerts:
        st.success("All detected categories are above their configured stock thresholds.")
        return

    for alert in alerts:
        st.error(alert)


def render_gap_analysis_tab(results: dict[str, Any]) -> None:
    """Render gap analytics with strong visual emphasis."""
    gap_count = results["gap_count"]

    if gap_count == 0:
        gap_message = "Shelf spacing looks healthy across adjacent detections."
        st.markdown(
            f"""
            <div class="gap-emphasis">
                <span class="gap-label">Gap analysis</span>
                <span class="gap-value">{gap_count}</span>
                <p class="gap-copy">{gap_message}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.success("No significant shelf gaps detected.")
        return

    gap_message = "Visible horizontal gaps suggest potential replenishment or planogram attention."
    st.markdown(
        f"""
        <div class="gap-emphasis">
            <span class="gap-label">Gap analysis</span>
            <span class="gap-value">{gap_count}</span>
            <p class="gap-copy">{gap_message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.warning("Gap count is above zero. Review the shelf image for facings that may need action.")


def render_analytics_section(results: dict[str, Any], controls: DashboardControls) -> None:
    """Render the tabbed analytics area below the primary dashboard layout."""
    if not controls.show_analytics:
        with st.container(border=True):
            render_panel_heading("🧠", "Analytics hidden", "Display preferences")
            st.info("Enable 'Show analytics' in the sidebar to view category, alert, and gap insights.")
        return

    with st.container(border=True):
        render_panel_heading("📈", "Detailed Analytics", "Deeper shelf intelligence")
        tab_1, tab_2, tab_3 = st.tabs(
            ["Category Breakdown", "Stock Alerts", "Gap Analysis"]
        )

        with tab_1:
            render_category_breakdown_tab(results)
        with tab_2:
            render_alerts_tab(results)
        with tab_3:
            render_gap_analysis_tab(results)


def render_dashboard(results: dict[str, Any], controls: DashboardControls) -> None:
    """Render the main dashboard layout once processing is complete."""
    render_analysis_meta(results, controls)

    left_col, right_col = st.columns([1.45, 1], gap="large")
    with left_col:
        render_image_panel(results, controls)
    with right_col:
        render_summary_panel(results, controls)

    st.markdown("")
    render_analytics_section(results, controls)


def main() -> None:
    """Build the premium Streamlit UI and analyze an uploaded shelf image."""
    st.set_page_config(
        page_title="Retail Shelf Intelligence System",
        page_icon="🛒",
        layout="wide",
    )
    inject_custom_css()
    render_header()

    uploaded_file, controls = render_sidebar()

    if uploaded_file is None:
        render_empty_state()
        return

    file_bytes = uploaded_file.getvalue()

    try:
        results, processed_now = get_or_process_results(
            file_bytes=file_bytes,
            filename=uploaded_file.name,
            controls=controls,
        )

        if processed_now:
            st.success("✅ Analysis complete. Insights ready.")

        render_dashboard(results, controls)
    except ImportError as error:
        st.error(f"Dependency error: {error}")
    except Exception as error:  # pragma: no cover - UI safeguard
        st.error(f"Unable to analyze the uploaded image: {error}")


if __name__ == "__main__":
    main()
