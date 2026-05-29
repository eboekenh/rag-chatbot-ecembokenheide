"""Diagnose: what is actually on the page after full load."""
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import os

os.makedirs("screenshots_ollama", exist_ok=True)

options = webdriver.ChromeOptions()
options.add_argument("--headless=new")
options.add_argument("--window-size=1600,1200")
driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)

driver.get("http://localhost:8503")
print("Page title:", driver.title)

for wait in [5, 10, 20, 30]:
    time.sleep(5)
    elapsed = wait
    textareas = driver.find_elements(By.TAG_NAME, "textarea")
    inputs = driver.find_elements(By.TAG_NAME, "input")
    testids = [el.get_attribute("data-testid") for el in driver.find_elements(By.CSS_SELECTOR, "[data-testid]")]
    print(f"\n--- After {elapsed}s ---")
    print(f"  textareas: {len(textareas)}")
    print(f"  inputs: {len(inputs)}")
    print(f"  data-testid values: {testids[:20]}")
    if textareas:
        print("  FOUND textarea!")
        break

driver.save_screenshot("screenshots_ollama/debug_final.png")
print("\nScreenshot: screenshots_ollama/debug_final.png")

# Print page source (ascii-safe)
src = driver.page_source.encode("ascii", "replace").decode("ascii")
print("\n--- Page source (first 2000 chars) ---")
print(src[:2000])

driver.quit()
