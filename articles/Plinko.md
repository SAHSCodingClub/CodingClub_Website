<div class="heading" style="text-align: center; font-weight: 700; font-size: 3rem;">
Casino Plinko Strategy
</div>

*By Logan Rembisz | December 18, 2024*

## DISCLAIMER

All statistics were gathered from a ROBLOX variation of plinko. I do not support or promote gambling. This article is for entertainment purposes only.

## Introduction

Plinko is a casino game in where you drop a ball and it bounces its way down until it ends up in a slot. Based on the slot in which it lands you earn a different percentage of your money back.

<img src="./images/plinko.png" alt="image of a ROBLOX plinko board" style="max-width: 60vw; height: auto;"/>

While playing plinko you are always guarenteed to get some money back, even if it is only 20% of what you originally bet. For example if you bet $100 and the ball lands in the 0.2 zone you would still be left with $20.

## Bet Sizes

When playing plinko professionally you always want to keep balls rolling down the board. In order to do this you always want to be betting 1/30th of your current balance. For example if you have $30,000 you should be betting $1,000. It should remain like this until you get $300,000 where you can start betting $10,000.

## Auto Clicker

In order to maximize my playtime while not hurting my fingers I developed a simple python program in order to click the button for me. It also allows me to pause the program so I can increase my bet sizes.

```cpp
import time
import pyautogui
import keyboard

while not keyboard.is_pressed("h"):
    print("press r to start")
    keyboard.wait("r")

    print("press e to stop")

    while not keyboard.is_pressed("e"):
        pyautogui.click()
        time.sleep(0.001)

print("clicking ended")
```
