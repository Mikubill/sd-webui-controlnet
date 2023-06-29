import argparse
import unittest
import os
import sys
import time
import datetime
from enum import Enum

import cv2
import requests
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


TIMEOUT = 20  # seconds
CWD = os.getcwd()
SKI_IMAGE = os.path.join(CWD, "images/ski.jpg")

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
test_result_dir = os.path.join("results", f"test_result_{timestamp}")
test_expectation_dir = "expectations"
os.makedirs(test_result_dir, exist_ok=True)
os.makedirs(test_expectation_dir, exist_ok=True)


class GenType(Enum):
    txt2img = "txt2img"
    img2img = "img2img"

    def _find_by_xpath(self, driver: webdriver.Chrome, xpath: str) -> "WebElement":
        return driver.find_element(By.XPATH, xpath)

    def tab(self, driver: webdriver.Chrome) -> "WebElement":
        return self._find_by_xpath(
            driver,
            f"//*[@id='tabs']/*[contains(@class, 'tab-nav')]//button[text()='{self.value}']",
        )

    def controlnet_panel(self, driver: webdriver.Chrome) -> "WebElement":
        return self._find_by_xpath(
            driver, f"//*[@id='tab_{self.value}']//*[@id='controlnet']"
        )

    def generate_button(self, driver: webdriver.Chrome) -> "WebElement":
        return self._find_by_xpath(driver, f"//*[@id='{self.value}_generate_box']")


class SeleniumTestCase(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.driver = webdriver.Chrome(ChromeDriverManager().install())
        self.gen_type = None

    def setUp(self) -> None:
        super().setUp()
        self.driver.get(webui_url)
        wait = WebDriverWait(self.driver, TIMEOUT)
        wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "#controlnet")))
        self.gen_type = GenType.txt2img

    def tearDown(self) -> None:
        self.driver.quit()
        super().tearDown()

    def select_gen_type(self, gen_type: GenType):
        gen_type.tab(self.driver).click()
        self.gen_type = gen_type

    def expand_controlnet_panel(self):
        controlnet_panel = self.gen_type.controlnet_panel(self.driver)
        input_image_group = controlnet_panel.find_element(
            By.CSS_SELECTOR, ".cnet-input-image-group"
        )
        if not input_image_group.is_displayed():
            controlnet_panel.click()

    def select_control_type(self, control_type: str):
        controlnet_panel = self.gen_type.controlnet_panel(self.driver)
        control_type_radio = controlnet_panel.find_element(
            By.CSS_SELECTOR, f'.controlnet_control_type input[value="{control_type}"]'
        )
        control_type_radio.click()
        time.sleep(3)  # Wait for gradio backend to update model/module

    def set_seed(self, seed: int):
        seed_input = self.driver.find_element(
            By.CSS_SELECTOR, f"#{self.gen_type.value}_seed input[type='number']"
        )
        seed_input.clear()
        seed_input.send_keys(seed)

    def set_subseed(self, seed: int):
        show_button = self.driver.find_element(
            By.CSS_SELECTOR,
            f"#{self.gen_type.value}_subseed_show input[type='checkbox']",
        )
        if not show_button.is_selected():
            show_button.click()

        subseed_locator = (
            By.CSS_SELECTOR,
            f"#{self.gen_type.value}_subseed input[type='number']",
        )
        WebDriverWait(self.driver, TIMEOUT).until(
            EC.visibility_of_element_located(subseed_locator)
        )
        subseed_input = self.driver.find_element(*subseed_locator)
        subseed_input.clear()
        subseed_input.send_keys(seed)

    def upload_controlnet_input(self, img_path: str):
        controlnet_panel = self.gen_type.controlnet_panel(self.driver)
        image_input = controlnet_panel.find_element(
            By.CSS_SELECTOR, '.cnet-input-image-group .cnet-image input[type="file"]'
        )
        image_input.send_keys(img_path)

    def generate_image(self, name: str):
        self.gen_type.generate_button(self.driver).click()
        progress_bar_locator_visible = EC.visibility_of_element_located(
            (By.CSS_SELECTOR, f"#{self.gen_type.value}_results .progress")
        )
        WebDriverWait(self.driver, TIMEOUT).until(progress_bar_locator_visible)
        WebDriverWait(self.driver, TIMEOUT * 10).until_not(progress_bar_locator_visible)
        generated_imgs = self.driver.find_elements(
            By.CSS_SELECTOR,
            f"#{self.gen_type.value}_results #{self.gen_type.value}_gallery img",
        )
        for i, generated_img in enumerate(generated_imgs):
            # Use requests to get the image content
            img_content = requests.get(generated_img.get_attribute("src")).content

            # Save the image content to a file
            global overwrite_expectation
            dest_dir = (
                test_expectation_dir if overwrite_expectation else test_result_dir
            )
            img_file_name = f"{self.__class__.__name__}_{name}_{i}.png"
            with open(
                os.path.join(dest_dir, img_file_name),
                "wb",
            ) as img_file:
                img_file.write(img_content)

            if not overwrite_expectation:
                img1 = cv2.imread(os.path.join(test_expectation_dir, img_file_name))
                img2 = cv2.imread(os.path.join(test_result_dir, img_file_name))
                self.expect_same_image(
                    img1,
                    img2,
                    diff_img_path=os.path.join(
                        test_result_dir, img_file_name.replace(".png", "_diff.png")
                    ),
                )

    def expect_same_image(self, img1, img2, diff_img_path: str):
        # Calculate the difference between the two images
        diff = cv2.absdiff(img1, img2)

        # Set a threshold to highlight the different pixels
        threshold = 30
        diff_highlighted = np.where(diff > threshold, 255, 0).astype(np.uint8)

        # Assert that the two images are similar within a tolerance
        similar = np.allclose(img1, img2, rtol=1e-05, atol=1e-08)
        if not similar:
            # Save the diff_highlighted image to inspect the differences
            cv2.imwrite(diff_img_path, diff_highlighted)

        self.assertTrue(similar)


class SeleniumTxt2ImgTest(SeleniumTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.select_gen_type(GenType.txt2img)
        self.set_seed(100)
        self.set_subseed(1000)

    def test_simple_control_types(self):
        """Test simple control types that only requires a reference image."""
        simple_control_types = {
            "Canny": "canny",
            "Depth": "depth_midas",
            "Normal": "normal_bae",
            "OpenPose": "openpose_full",
            "MLSD": "mlsd",
            "Lineart": "lineart_standard (from white bg & black line)",
            "SoftEdge": "softedge_pidinet",
            "Scribble": "scribble_pidinet",
            "Seg": "seg_ofade20k",
            # Shuffle is currently non-deterministic
            # "Shuffle": "shuffle",
            "Tile": "tile_resample", 
            "Reference": "reference_only",
        }.keys()

        for control_type in simple_control_types:
            with self.subTest(control_type=control_type):
                self.expand_controlnet_panel()
                self.select_control_type(control_type)
                self.upload_controlnet_input(SKI_IMAGE)
                self.generate_image(f"{control_type}_ski")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your script description.")
    parser.add_argument(
        "--overwrite_expectation", action="store_true", help="overwrite expectation"
    )
    parser.add_argument(
        "--target_url", type=str, default="http://localhost:7860", help="WebUI URL"
    )
    args = parser.parse_args()
    overwrite_expectation = args.overwrite_expectation
    webui_url = args.target_url

    sys.argv = sys.argv[:1]
    unittest.main()
