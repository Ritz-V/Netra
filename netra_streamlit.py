from __future__ import annotations
from typing import List

import cv2
import streamlit as st

from netra.models import Event
from netra.pipeline import run_stream

st.set_page_config(
    page_title="Netra – Behavioural Safety POC",
    layout="wide",
)


def main() -> None:
    st.title("Netra – Behavioural Safety POC")
    st.markdown(
        """
        **Netra** is an AI-powered behavioural safety prototype for public spaces.

        - Uses CCTV/video + computer vision.
        - Detects behavioural **patterns only** – *no faces, no identity*.
        - Modules: violence, crowd panic, falls, child distress, abandoned objects.
        """
    )

    with st.sidebar:
        st.header("Input Source")
        source_type = st.radio(
            "Choose source",
            ["Demo video file", "Webcam"],
            index=0,
        )

        if source_type == "Demo video file":
            video_file = st.text_input(
                "Path to video file (e.g. ./samples/crowd.mp4)",
                value="",
            )
        else:
            cam_index = st.number_input(
                "Webcam index", min_value=0, max_value=4, value=0, step=1
            )

        st.header("Modules")
        enable_violence = st.checkbox("Violence & aggression", value=True)
        enable_crowd = st.checkbox("Crowd panic / stampede", value=True)
        enable_fall = st.checkbox("Fall / collapse", value=True)
        enable_child = st.checkbox("Child distress / forced movement", value=True)
        enable_abandoned = st.checkbox("Abandoned object", value=True)

        run_button = st.button("Start Analysis")

    col_video, col_right = st.columns([2, 1])

    with col_video:
        video_placeholder = st.empty()

    with col_right:
        status_placeholder = st.empty()
        incidents_placeholder = st.empty()

    if "incident_log" not in st.session_state:
        st.session_state["incident_log"] = []

    if not run_button:
        st.info("Select input source and click *Start Analysis* to begin.")
        return

    if source_type == "Demo video file":
        if not video_file:
            st.error("Please enter a valid video file path.")
            return
        source = video_file
    else:
        source = int(cam_index)

    enable_modules = {
        "violence": enable_violence,
        "crowd_panic": enable_crowd,
        "fall": enable_fall,
        "child_distress": enable_child,
        "abandoned": enable_abandoned,
    }

    st.success("Starting analysis… First YOLOv5 load may take a moment.")

    try:
        for frame_index, frame_bgr, events, status in run_stream(
            source=source, enable_modules=enable_modules
        ):
            # Update incident log
            for e in events:
                st.session_state["incident_log"].append(e)

            log: List[Event] = st.session_state["incident_log"][-50:]

            # Convert frame to RGB for display
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            video_placeholder.image(
                frame_rgb,
                channels="RGB",
                caption=f"Frame {frame_index} | Severity {status.overall_severity} ({status.color})",
                use_column_width=True,
            )

            with status_placeholder.container():
                st.subheader("Live Status")
                st.metric("Current severity", status.overall_severity)
                st.write(f"Status colour: **{status.color.upper()}**")
                if status.active_events:
                    st.write("Active events this frame:")
                    for e in status.active_events:
                        st.write(f"- [{e.event_type.value}] Sev {e.severity}: {e.description}")
                else:
                    st.write("No active high-risk events detected in this frame.")

            with incidents_placeholder.container():
                st.subheader("Recent incidents")
                if not log:
                    st.write("No incidents logged yet.")
                else:
                    for e in reversed(log[-10:]):
                        st.write(
                            f"Frame {e.frame_index} | {e.event_type.value} | Severity {e.severity}"
                        )
                        st.caption(e.description)

    except RuntimeError as exc:
        st.error(str(exc))


if __name__ == "__main__":
    main()
