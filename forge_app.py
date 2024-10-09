from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.dynamic_module_utils import get_imports
from PIL import Image, ImageDraw, UnidentifiedImageError
from unittest.mock import patch
from io import BytesIO
from glob import glob
from tqdm import tqdm

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import gradio as gr
import numpy as np
import random
import copy
import json
import os

import spaces


COLOR_MAP: tuple[str] = (
    "blue",
    "orange",
    "green",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
    "red",
    "lime",
    "indigo",
    "violet",
    "aqua",
    "magenta",
    "coral",
    "gold",
    "tan",
    "skyblue",
)

single_task_list: tuple[str] = (
    "Caption",
    "Detailed Caption",
    "More Detailed Caption",
    "Object Detection",
    "Dense Region Caption",
    "Region Proposal",
    "Caption to Phrase Grounding",
    "Referring Expression Segmentation",
    "Region to Segmentation",
    "Open Vocabulary Detection",
    "Region to Category",
    "Region to Description",
    "OCR",
    "OCR with Region",
)

cascased_tasks_list: tuple[str] = (
    "Caption + Grounding",
    "Detailed Caption + Grounding",
    "More Detailed Caption + Grounding",
)


def _fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports


with spaces.capture_gpu_object() as GO:

    with patch("transformers.dynamic_module_utils.get_imports", _fixed_get_imports):
        models: dict[str, AutoModelForCausalLM] = {
            "microsoft/Florence-2-large": AutoModelForCausalLM.from_pretrained(
                "microsoft/Florence-2-large", trust_remote_code=True
            )
            .to("cuda")
            .eval(),
        }

    processors: dict[str, AutoProcessor] = {
        "microsoft/Florence-2-large": AutoProcessor.from_pretrained(
            "microsoft/Florence-2-large", trust_remote_code=True
        ),
    }


class Utils:

    @staticmethod
    def fig_to_pil(fig):
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        return Image.open(buf)

    @staticmethod
    def plot_bbox(image, data):
        fig, ax = plt.subplots()
        ax.imshow(image)
        for bbox, label in zip(data["bboxes"], data["labels"]):
            x1, y1, x2, y2 = bbox
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="r", facecolor="none"
            )
            ax.add_patch(rect)
            plt.text(
                x1,
                y1,
                label,
                color="white",
                fontsize=8,
                bbox=dict(facecolor="red", alpha=0.5),
            )
        ax.axis("off")
        return fig

    @staticmethod
    def draw_polygons(image, prediction, fill_mask=False):
        draw = ImageDraw.Draw(image)
        scale = 1
        for polygons, label in zip(prediction["polygons"], prediction["labels"]):
            color = random.choice(COLOR_MAP)
            fill_color = random.choice(COLOR_MAP) if fill_mask else None
            for _polygon in polygons:
                _polygon = np.array(_polygon).reshape(-1, 2)
                if len(_polygon) < 3:
                    print("Invalid polygon:", _polygon)
                    continue
                _polygon = (_polygon * scale).reshape(-1).tolist()
                if fill_mask:
                    draw.polygon(_polygon, outline=color, fill=fill_color)
                else:
                    draw.polygon(_polygon, outline=color)
                draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color)
        return image

    @staticmethod
    def convert_to_od_format(data):
        bboxes = data.get("bboxes", [])
        labels = data.get("bboxes_labels", [])
        od_results = {"bboxes": bboxes, "labels": labels}
        return od_results

    @staticmethod
    def draw_ocr_bboxes(image, prediction):
        scale = 1
        draw = ImageDraw.Draw(image)
        bboxes, labels = prediction["quad_boxes"], prediction["labels"]
        for box, label in zip(bboxes, labels):
            color = random.choice(COLOR_MAP)
            new_box = (np.array(box) * scale).tolist()
            draw.polygon(new_box, width=3, outline=color)
            draw.text(
                (new_box[0] + 8, new_box[1] + 2),
                "{}".format(label),
                align="right",
                fill=color,
            )
        return image


def run_example(
    task_prompt: str,
    image: Image.Image,
    text_input: str = None,
    model_id: str = "microsoft/Florence-2-large",
):
    model = models[model_id]
    processor = processors[model_id]
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, task=task_prompt, image_size=(image.width, image.height)
    )
    return parsed_answer


def _process_image(
    image: Image.Image,
    task_type: str,
    text_input: str,
    model_id: str,
) -> tuple[dict, Image.Image]:

    match task_type:
        case "Caption":
            task = "<CAPTION>"
            results = run_example(task, image, model_id=model_id)
            return results, None

        case "Detailed Caption":
            task = "<DETAILED_CAPTION>"
            results = run_example(task, image, model_id=model_id)
            return results, None

        case "More Detailed Caption":
            task = "<MORE_DETAILED_CAPTION>"
            results = run_example(task, image, model_id=model_id)
            return results, None

        case "Caption + Grounding":
            task = "<CAPTION>"
            results = run_example(task, image, model_id=model_id)
            text_input = results[task]
            task = "<CAPTION_TO_PHRASE_GROUNDING>"
            results = run_example(task, image, text_input, model_id)
            results["<CAPTION>"] = text_input
            fig = Utils.plot_bbox(image, results["<CAPTION_TO_PHRASE_GROUNDING>"])
            return results, Utils.fig_to_pil(fig)

        case "Detailed Caption + Grounding":
            task = "<DETAILED_CAPTION>"
            results = run_example(task, image, model_id=model_id)
            text_input = results[task]
            task = "<CAPTION_TO_PHRASE_GROUNDING>"
            results = run_example(task, image, text_input, model_id)
            results["<DETAILED_CAPTION>"] = text_input
            fig = Utils.plot_bbox(image, results["<CAPTION_TO_PHRASE_GROUNDING>"])
            return results, Utils.fig_to_pil(fig)

        case "More Detailed Caption + Grounding":
            task = "<MORE_DETAILED_CAPTION>"
            results = run_example(task, image, model_id=model_id)
            text_input = results[task]
            task = "<CAPTION_TO_PHRASE_GROUNDING>"
            results = run_example(task, image, text_input, model_id)
            results["<MORE_DETAILED_CAPTION>"] = text_input
            fig = Utils.plot_bbox(image, results["<CAPTION_TO_PHRASE_GROUNDING>"])
            return results, Utils.fig_to_pil(fig)

        case "Object Detection":
            task = "<OD>"
            results = run_example(task, image, model_id=model_id)
            fig = Utils.plot_bbox(image, results["<OD>"])
            return results, Utils.fig_to_pil(fig)

        case "Dense Region Caption":
            task = "<DENSE_REGION_CAPTION>"
            results = run_example(task, image, model_id=model_id)
            fig = Utils.plot_bbox(image, results["<DENSE_REGION_CAPTION>"])
            return results, Utils.fig_to_pil(fig)

        case "Region Proposal":
            task = "<REGION_PROPOSAL>"
            results = run_example(task, image, model_id=model_id)
            fig = Utils.plot_bbox(image, results["<REGION_PROPOSAL>"])
            return results, Utils.fig_to_pil(fig)

        case "Caption to Phrase Grounding":
            task = "<CAPTION_TO_PHRASE_GROUNDING>"
            results = run_example(task, image, text_input, model_id)
            fig = Utils.plot_bbox(image, results["<CAPTION_TO_PHRASE_GROUNDING>"])
            return results, Utils.fig_to_pil(fig)

        case "Referring Expression Segmentation":
            task = "<REFERRING_EXPRESSION_SEGMENTATION>"
            results = run_example(task, image, text_input, model_id)
            output_image = copy.deepcopy(image)
            output_image = Utils.draw_polygons(
                output_image,
                results["<REFERRING_EXPRESSION_SEGMENTATION>"],
                fill_mask=True,
            )
            return results, output_image

        case "Region to Segmentation":
            task = "<REGION_TO_SEGMENTATION>"
            results = run_example(task, image, text_input, model_id)
            output_image = copy.deepcopy(image)
            output_image = Utils.draw_polygons(
                output_image, results["<REGION_TO_SEGMENTATION>"], fill_mask=True
            )
            return results, output_image

        case "Open Vocabulary Detection":
            task = "<OPEN_VOCABULARY_DETECTION>"
            results = run_example(task, image, text_input, model_id)
            bbox_results = Utils.convert_to_od_format(
                results["<OPEN_VOCABULARY_DETECTION>"]
            )
            fig = Utils.plot_bbox(image, bbox_results)
            return results, Utils.fig_to_pil(fig)

        case "Region to Category":
            task = "<REGION_TO_CATEGORY>"
            results = run_example(task, image, text_input, model_id)
            return results, None

        case "Region to Description":
            task = "<REGION_TO_DESCRIPTION>"
            results = run_example(task, image, text_input, model_id)
            return results, None

        case "OCR":
            task = "<OCR>"
            results = run_example(task, image, model_id=model_id)
            return results, None

        case "OCR with Region":
            task = "<OCR_WITH_REGION>"
            results = run_example(task, image, model_id=model_id)
            output_image = copy.deepcopy(image)
            output_image = Utils.draw_ocr_bboxes(
                output_image, results["<OCR_WITH_REGION>"]
            )
            return results, output_image

        case _:
            raise gr.Error("Unrecognized Task...")


def _grab_images(folder: str, recursive: bool) -> dict[str, Image.Image]:
    images = {}
    files = (
        glob(os.path.join(folder, "**", "*"), recursive=True)
        if recursive
        else [os.path.join(folder, f) for f in os.listdir(folder)]
    )

    for file in files:
        if os.path.isdir(file) or file.endswith("_bbox.png"):
            continue
        try:
            image = Image.open(file)
            images.update({file: image})
        except UnidentifiedImageError:
            pass

    return images


@spaces.GPU(gpu_objects=[GO], manual_load=False)
def process_image(i_dir: str, recursive: bool, task: str, text: str, mdl: str):
    if not os.path.isdir(i_dir):
        raise gr.Error(f'Path "{i_dir}" is not a folder...')

    images = _grab_images(i_dir, recursive)

    if len(images.keys()) == 0:
        raise gr.Error(f'Folder "{i_dir}" is empty...')

    for filename, image in tqdm(images.items()):
        file, ext = os.path.splitext(filename)
        res, img = _process_image(image, task, text, mdl)

        flag = False
        if isinstance(res, dict):
            values = list(res.values())
            if len(values) == 1 and isinstance(values[0], str):
                with open(f"{file}.txt", "w+", encoding="utf-8") as f:
                    f.write(values[0])
                    flag = True
        if not flag:
            with open(f"{file}.json", "w+", encoding="utf-8") as f:
                json.dump(res, f)

        if img:
            img.save(f"{file}_bbox.png", optimize=True)


with gr.Blocks(analytics_enabled=False).queue() as demo:
    gr.HTML(
        """<h1 align="center">
        <a href="https://huggingface.co/microsoft/Florence-2-large">Florence-2</a>
        </h1>"""
    )

    with gr.Row(variant="panel"):
        with gr.Column(variant="compact"):
            florence: str = "microsoft/Florence-2-large"
            model_selector = gr.Dropdown(
                label="Model",
                choices=[florence],
                value=florence,
                interactive=False,
            )

            task_type = gr.Radio(
                label="Task Type",
                choices=("Single Task", "Cascased Tasks"),
                value="Single Task",
            )

            task_selector = gr.Dropdown(
                label="Task",
                choices=single_task_list,
                value="More Detailed Caption",
            )

            text_input = gr.Textbox(
                label="Additional Prompt",
                lines=3,
                max_lines=3,
            )

        with gr.Column(variant="compact"):
            input_dir = gr.Textbox(label="Working Directory", lines=1, max_lines=1)
            with gr.Row(variant="panel"):
                run_btn = gr.Button(value="Run", variant="primary", scale=2)
                recursive = gr.Checkbox(True, label="Recursive", scale=1)

    def _update_tasks(choice: str):
        if choice == "Single Task":
            return gr.Dropdown(choices=single_task_list, value="More Detailed Caption")
        else:
            return gr.Dropdown(choices=cascased_tasks_list, value="Caption + Grounding")

    task_type.change(
        fn=_update_tasks,
        inputs=[task_type],
        outputs=[task_selector],
        show_progress="hidden",
    )

    run_btn.click(
        fn=process_image,
        inputs=[
            input_dir,
            recursive,
            task_selector,
            text_input,
            model_selector,
        ],
    )


if __name__ == "__main__":
    demo.launch()
