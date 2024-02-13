import gradio as gr
from main import predict_slate_health, add_key

# Adding custom CSS styles
custom_css = """
#heading {
    background-color: #b5f1f5;
    }
    
.custom-output {
  font-weight: 800;
  background-color: #f2b8c6;
  font-size: 25px;
  color: #f2b8c6;
}
.label-bg {
    background-color: #cbd4d3; /* Background color for labels */
    padding: 5px; /* Adding padding for labels */
}
.custom-class {
    background-color: #f2f2f2;
    border: 1px solid #ccc;
    padding: 10px;
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Label("eSlates Classification", elem_id="heading", elem_classes=["label-bg"])
    with gr.Row():
        gr.Label("Overall Rating", elem_classes=["custom-output"])
        overall_rating = gr.Textbox(
            show_label=False, lines=3, elem_classes=["custom-output"]
        )
        gr.Label("Remarks", elem_classes=["custom-output"], container=True)
        overall_reason = gr.Textbox(
            show_label=False, lines=3, elem_classes=["custom-output"]
        )
    with gr.Row():
        with gr.Column():
            gr.Label("Demographic", elem_classes=["label-bg"])
            with gr.Group():  # Grouping input text boxes
                urm = gr.Textbox(label="URM", value="15")
                minority = gr.Textbox(label="Minority", value="50")
                female = gr.Textbox(label="Female", value="40")
            gr.Label("Geolocation", elem_classes=["label-bg"])
            with gr.Group():
                ea = gr.Textbox(label="EA", value="30")
                so = gr.Textbox(label="SO", value="30")
                ce = gr.Textbox(label="CE", value="15")
                we = gr.Textbox(label="WE", value="25")
                fo = gr.Textbox(label="FO", value="5")
            gr.Label("Seniority", elem_classes=["label-bg"])
            with gr.Group():
                prof = gr.Textbox(label="Professor", value="70")
                assoc_prof = gr.Textbox(label="Associate Professor", value="20")
                ass_prof = gr.Textbox(label="Assistant Professor", value="10")
            workflow = gr.TextArea(label="Workflow")
            sub_button = gr.Button("Submit")
        with gr.Column():
            with gr.Row():
                gr.Label("Average Rating", elem_classes=["label-bg"])
                avg_rating_demo = gr.Textbox(label="")
            with gr.Group():
                with gr.Row():
                    urm_rating = gr.Textbox(label="Rating")
                    urm_reason = gr.Textbox(label="Reason")
                with gr.Row():
                    minority_rating = gr.Textbox(label="Rating")
                    minority_reason = gr.Textbox(label="Reason")
                with gr.Row():
                    female_rating = gr.Textbox(label="Rating")
                    female_reason = gr.Textbox(label="Reason")
            with gr.Row():
                gr.Label("Average Rating", elem_classes=["label-bg"])
                avg_rating_geo = gr.Textbox(label="")
            with gr.Group():
                with gr.Row():
                    ea_rating = gr.Textbox(label="Rating")
                    ea_reason = gr.Textbox(label="Reason")
                with gr.Row():
                    so_rating = gr.Textbox(label="Rating")
                    so_reason = gr.Textbox(label="Reason")
                with gr.Row():
                    ce_rating = gr.Textbox(label="Rating")
                    ce_reason = gr.Textbox(label="Reason")
                with gr.Row():
                    we_rating = gr.Textbox(label="Rating")
                    we_reason = gr.Textbox(label="Reason")
                with gr.Row():
                    fo_rating = gr.Textbox(label="Rating")
                    fo_reason = gr.Textbox(label="Reason")
            with gr.Row():
                gr.Label("Average Rating", elem_classes=["label-bg"])
                avg_rating_career = gr.Textbox(label="")
            with gr.Group():
                with gr.Row():
                    prof_rating = gr.Textbox(label="Rating")
                    prof_reason = gr.Textbox(label="Reason")
                with gr.Row():
                    assoc_prof_rating = gr.Textbox(label="Rating")
                    assoc_prof_reason = gr.Textbox(label="Reason")
                with gr.Row():
                    ass_prof_rating = gr.Textbox(label="Rating")
                    ass_prof_reason = gr.Textbox(label="Reason")

    sub_button.click(
        fn=predict_slate_health,
        inputs=[
            urm,
            minority,
            female,
            ea,
            so,
            ce,
            we,
            fo,
            prof,
            assoc_prof,
            ass_prof,
            workflow,
        ],
        outputs=[
            overall_rating,
            overall_reason,
            avg_rating_demo,
            urm_rating,
            urm_reason,
            minority_rating,
            minority_reason,
            female_rating,
            female_reason,
            avg_rating_geo,
            ea_rating,
            ea_reason,
            so_rating,
            so_reason,
            ce_rating,
            ce_reason,
            we_rating,
            we_reason,
            fo_rating,
            fo_reason,
            avg_rating_career,
            prof_rating,
            prof_reason,
            assoc_prof_rating,
            assoc_prof_reason,
            ass_prof_rating,
            ass_prof_reason,
        ],
        scroll_to_output=True,
    )

with gr.Blocks(css=custom_css) as tab2:
    gr.Label("Add OpenAI Key", elem_id="heading", elem_classes=["label-bg"])

    with gr.Group():
        key = gr.Textbox(label="OPENAI Key")
        deployment = gr.Textbox(label="DEPLOYMENT")
        endpoint = gr.Textbox(
            label="AZURE ENDPOINT"
        )

        with gr.Row():
            sub_button2 = gr.Button("Submit")
            gr.ClearButton(key, deployment, endpoint)

    sub_button2.click(fn=add_key, inputs=[key, deployment, endpoint])
tabs = gr.TabbedInterface(
    [tab2, demo], ["Add Key", "eSlate-classification"], css=custom_css
)

if __name__ == "__main__":
    tabs.launch()
# demo.launch(share=True)
