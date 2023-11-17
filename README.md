*** Capturing and Analyzing Semantic Variance in Image Annotations ***

Welcome to a unique project aimed at diving into the intricate world of image annotations and their semantic variance. The primary focus here is the extraction, processing, and insightful analysis of text annotations linked with a diverse collection of images. Leveraging advanced techniques such as tokenization, stopwords removal, part-of-speech tagging, and synset generation, the goal is to scrutinize the semantic variances within these annotations. The results of this project include similarity scores, reflecting the semantic consistency of the annotations, and visual representations to help articulate the extent of variance in the annotations.

70 images were generated in https://openai.com/dall-e-2 by typing in various descriptions. Each image has at least one theme overlapping with another image. 

The images were then added to google forms and annotated by a group of peers. Their annotations were then processed in this project. The results were able to determine which images were the most clearly identified and which were not.

*** Prerequisites ***

This project is developed in Python and leverages several Python libraries, including pandas, NLTK, matplotlib, and numpy. All these libraries are managed and installed using pip, Python's renowned package installer. If you haven't installed pip yet, follow the instructions available here: https://pip.pypa.io/en/stable/installation/ 

Post pip installation, ensure it's updated to the latest version by executing the following command:
```pip install --upgrade pip```

*** Setup ***

To avoid any conflicts with other Python projects or system-wide packages, it's recommended to use a virtual environment. 
Here’s how to set it up:

Navigate to the project's root directory in your terminal and run:
```python3 -m venv venv```

This command creates a new virtual environment named venv in your project directory.

Then, activate the virtual environment.
  On Windows: ```.\venv\Scripts\activate```
  On Mac/Linux:  ```source venv/bin/activate```

To install all the required Python libraries for this project, use the command below:
```pip install -r requirements.txt```

This command will initiate the installation of all necessary libraries listed in the `requirements.txt` file.

*** Project Structure ***

The project consists of several key components:

Images: This folder is the repository of all images used in this project.
Annotations: This folder holds all the annotations that are processed during the project. These annotations are contained within CSV files and represent textual descriptions of the images.
Scripts: Each script has its unique main method and is designed to run individually, fulfilling a specific purpose in the semantic analysis of the annotations.

*** Running the Scripts ***

To run the scripts, run : ```python main.py```

This will bring up a menu in the terminal that provides the options to run the different scripts.

*** Outputs ***

Each script outputs specific results contributing to the overall analysis of semantic similarity. Expect similarity scores for the annotations and graphical representations that visually present the semantic variance within the image annotations.

By interpreting these outputs, you can identify images based on their similarity scores, pinpoint images with the most and least variance in their annotations, and calculate average similarity scores for predefined categories such as "Animals", "Setting", "Colors", etc. 

Due to the amount of keywords and images, the graph does not display the images but rather vertices that are numbered representing each image. To see the images in the graph, run the demo version of the graph. This displays a small amount of randomly selected images and their respective keywords.

It is recommended with all of the graphs to enlarge to page to see all that it has to offer. Feel free to zoom in and browse around at all of the keywords. 

When you are finished, simply exit the graph.

Enjoy!