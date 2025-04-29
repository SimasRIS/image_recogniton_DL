# Image Recognition Program (Using TensorFlow/Keras)

## What is this program?

This program learns to identify four different types of images that you store in the `img/` folder. Everything the program needs to do - from loading pictures to showing results - is in the **`image_recognition_DL.py`** file.

When using the normal settings (18 training rounds, 40 pictures at a time, with 20% of pictures used for testing), the program usually gets it right over 80% of the time. How well it works depends on having good, evenly distributed pictures.

---

## How to start using it

```
# 1. Get the program files
$ git clone https://github.com/SimasRIS/image_recogniton_DL.git
$ cd image_recogniton_DL

# 2. (Optional) set up a separate workspace
$ python -m venv venv
# macOS / Linux
$ source venv/bin/activate
# Windows
$ venv\Scripts\activate

# 3. Install needed software
$ pip install -r requirements.txt

# 4. Add your pictures
#    Put them in folders like this:
#    img/category_name/picture1.jpg
#    img/category_name/picture2.png

# 5. Start the program
$ python main.py
```

---

How to organize your pictures

- Put your pictures in folders like this: `img/category_name/...`
    - The program will use folder names to know what each picture is
    - You can use regular picture files like `.jpg`, `.jpeg`, or `.png`

Here's how your folders should look:

```
img/
├── car/
├── cat/
├── dog/
└── painting/
```

---

## How the program works

1. **Loading pictures** - The program takes all your pictures, mixes them up randomly, and makes them all the same size (**180 × 180 pixels**)
2. **Picture preparation** - The program adjusts the brightness of pictures to make them easier to process
3. **Program structure** - The program uses a simple design:
    - Takes in pictures
    - Processes them through two special filters
    - Converts pictures into useful information
    - Makes its final guess about what each picture shows
4. **Learning process** - The program practices identifying pictures 18 times, getting better each round
5. **Checking results** - After learning, the program shows you how well it did by:
    - showing the actual type of picture
    - showing what it thinks the picture is
    - showing how sure it is about its guess

---

## How to make it work better

- Add more pictures - the more examples the program has, the better it learns
    - Use picture editing tricks (like flipping or rotating) to create more training examples
    - Adjust settings like training time and picture batch size to improve accuracy