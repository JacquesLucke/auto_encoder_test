import tensorflow as tf
import numpy as np
import plotly.express as px
import tkinter
import PIL
import PIL.ImageTk
import PIL.Image

tot_images = 10

decoder: tf.keras.Model = tf.keras.models.load_model("decoder_model")
rng: np.random.Generator = np.random.default_rng()
latent_space = np.zeros(2)


def update(event=None):
    latent_space[0] = int(tk_slider1.get()) / 100
    latent_space[1] = int(tk_slider2.get()) / 100
    generated_image = decoder(np.array([latent_space]))[0]
    tk_image = PIL.ImageTk.PhotoImage(
        PIL.Image.fromarray((np.array(generated_image) + 1) * 128).resize((200, 200))
    )
    tk_panel.configure(image=tk_image)
    tk_panel.image = tk_image


tk_root = tkinter.Tk()
tk_root.title("Decoder Exploration")
tk_panel = tkinter.Label(tk_root, width=200, height=200)
tk_panel.pack()
tk_slider1 = tkinter.Scale(
    tk_root, from_=-100, to=100, orient=tkinter.HORIZONTAL, command=update
)
tk_slider1.pack()
tk_slider2 = tkinter.Scale(
    tk_root, from_=-100, to=100, orient=tkinter.HORIZONTAL, command=update
)
tk_slider2.pack()

update()

tk_root.mainloop()
