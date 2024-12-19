<div class="heading" style="text-align: center; font-weight: 700; font-size: 3rem;">
Beating Human Benchmark with Python Automation
</div>

*By Erlis Hoxha | December 19, 2024*

## Introduction

Human Benchmark is a website with various tests to determine your skill in various activities, like reaction time, number memory, and what my code focuses on: verbal memory.

<img src="./images/human_benchmark.png" alt="Human Benchmark verbal memory test" style="max-width: 60vw; height: auto;"/>

The verbal memory tests consists of storing a word in your short-term memory and determining if it was seen before or a new word. Average scores range from 10 words to 50 words. You start with three lives, and every time you incorrectly choose if the word is new or already appeared, you lose a life.

## Autocompletion

To prove that machines are superior, I created a simple code using Python that autocompletes the challenge. It uses mss, numpy, and pytesseract to take a screenshot and then process it into a string. From there, it will check if the string is already in the list of words, and click on the Seen button if it is. Otherwise, it will append the string to the list and then click the New button. So far, this code is 100% accurate.

```cpp
import pyautogui
import time
import pytesseract
import keyboard
import numpy as np
import mss

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'

#color = (149,195,232)

clicked = False

finished = False

box = {'top': 523, 'left': 14, 'width': 1856, 'height': 158}

seenPos = 850
newPos = 1054

prevWords = []

with mss.mss() as sct:  
    while True:
        key = keyboard.read_key()
        if key == 'q':
            finished = False
            while not finished:
                clicked = False
                if pyautogui.pixelMatchesColor(915,336,[255,255,255]):
                    finished = True
            
                im = np.asarray(sct.grab(box))
                word = pytesseract.image_to_string(im).lower()
                if word not in prevWords:
                    prevWords.append(word)
                    pyautogui.click(newPos,730)
                else:
                    pyautogui.click(seenPos,730)
                if key == 'e':
                    break
```
