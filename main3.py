from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN

from sklearn import metrics
y_pred = ["a", "b", "c", "a", "b"]
y_act = ["a", "b", "c", "c", "a"]
print(metrics.confusion_matrix(y_act, y_pred, labels=["a", "b", "c"]))
print(metrics.classification_report(y_act, y_pred, labels=["a",
"b","c"]))

def draw_image_with_boxes(filename, result_list):
    # load the image
    data = pyplot.imread(filename)
    # plot the image
    pyplot.imshow(data)
    # get the context for drawing boxes
    ax = pyplot.gca()
    # plot each box
    for result in result_list:
        # get coordinates
        x, y, width, height = result['box']
        # create the shape
        rect = Rectangle((x, y), width, height, fill=False, color='blue')
        # draw the box
        ax.add_patch(rect)
        # draw the dots
        #for key, value in result['keypoints'].items():
            # create and draw dot
           # dot = Circle(value, radius=2, color='red')
          #  ax.add_patch(dot)
    # show the plot
    pyplot.show()


filename = 'elon3.jpg'
pixels = pyplot.imread(filename)
# create the detector, using default weights
detector = MTCNN()
# detect faces in the image
faces = detector.detect_faces(pixels)
# display faces on the original image
draw_image_with_boxes(filename, faces)



