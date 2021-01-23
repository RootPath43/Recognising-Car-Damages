import glob
import os
import threading
from functools import partial
import cv2
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.filechooser import FileChooser
from kivy.uix.screenmanager import Screen, ScreenManager


class PhotoChoose(Screen):
    def imageDamage(self):
        file=open("Locations.txt",'r')
        path1= str(file.readline())
        print(path1)
        image1 = cv2.imread('./Car/TOYOTA_COROLLA_WHITE.jpg')
        dsize = (1920, 1080)
        output1 = cv2.resize(image1, dsize)
        print((path1))
        try:
            image2 = cv2.imread(path1)
            output2=cv2.resize(image2,dsize)
            difference = cv2.subtract(output1, output2)
            Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            output2[mask != 255] = [0, 0, 255]

            cv2.imshow("hello 2", output2)
        except:
            print("image is not found")


        cv2.waitKey(0)
        file.close()
        cv2.destroyAllWindows()

class ChooseScreen(Screen):
    pass
class DamageLive(Screen):
    def runner(self):
        threading.Thread(target=self.doit2, daemon=True).start()
    def doit2(self):
        self.do_vid = True  # flag to stop loop
        capture = cv2.VideoCapture(0)
        f=open("CarModel.txt",'r')
        carmodelpath=str(f.read())
        print(carmodelpath)
        dsize = (704, 480)
        imagetocompare="./"+carmodelpath+"/"+self.image
        print(imagetocompare)
        img1=cv2.imread(imagetocompare)
        output1=cv2.resize(img1,dsize)
        while (self.do_vid):
            ret, original = capture.read()
            output2=cv2.resize(original,dsize)
            difference = cv2.subtract(output1, output2)
            Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            output2[mask != 255] = [0, 0, 255]
            cv2.waitKey(3)
            Clock.schedule_once(partial(self.display_frame2, output2))
            cv2.putText(original, (carmodelpath+self.image),(20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, 2)
        capture.release()
        cv2.destroyAllWindows()

    def display_frame2(self, original, dt):
        # display the current video frame in the kivy Image widget

        # create a Texture the correct size and format for the frame
        texture = Texture.create(size=(original.shape[1], original.shape[0]), colorfmt='bgr')

        # copy the frame data into the texture
        texture.blit_buffer(original.tobytes(order=None), colorfmt='bgr', bufferfmt='ubyte')

        # flip the texture (otherwise the video is upside down
        texture.flip_vertical()

        # actually put the texture in the kivy Image widget
        self.ids.vid2.texture = texture

    def stop_vid(self):
        # stop the video capture loop
        self.do_vid = False
    def Image1(self):
        self.image="Back.jpg"
    def Image2(self):
        self.image="Front.jpg"
    def Image3(self):
        self.image="Left.jpg"
    def Image4(self):
        self.image="Right.jpg"
class FileChooser(Screen):
    firstimage=''
    def selected(self, path, filename):
        chosenImage=(str(filename)[1:-1])
        print(chosenImage)
        self.image1=chosenImage
        f=open("Locations.txt","w")
        f.write(chosenImage)
        f.close()
class CaptureScreen(Screen):
    def set_Label(self,decision_tuple)  :
        self.the_decision.text=str(decision_tuple)
class MainApp(App):
    def build(self):
        threading.Thread(target=self.doit, daemon=True).start()
        sm = ScreenManager()
        self.main_screen = CaptureScreen()
        sm.add_widget(self.main_screen)
        sm.add_widget(ChooseScreen(name='choose'))
        sm.add_widget(PhotoChoose(name='photochoose'))
        sm.add_widget(FileChooser(name='filechooser'))
        sm.add_widget(DamageLive(name='damagelive'))
        return sm

    def doit(self):
        number_keypoints = 0
        # this code is run in a separate thread
        self.do_vid = True  # flag to stop loop
        # make a window for use by cv2
        # flags allow resizing without regard to aspect ratio
        # resize the window to (0,0) to make it invisible
        capture = cv2.VideoCapture(0)
        all_images_to_compare = []
        titles = []
        car=[]
        percentage_similarity = []
        for f in glob.iglob("Car\*"):
            image = cv2.imread(f)
            titles.append((os.path.splitext(os.path.basename(f))[0]))
            all_images_to_compare.append(image)
        while(self.do_vid):
            ret, original = capture.read()
            sift = cv2.xfeatures2d.SIFT_create()
            index_params = dict(algorithm=0, trees=5)
            search_params = dict()
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            for image_to_compare, title in zip(all_images_to_compare, titles):

                kp_1, desc_1 = sift.detectAndCompute(original, None)
                kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)
                matches = flann.knnMatch(desc_1, desc_2, k=2)
                good_points = []
                number_keypoints = 0
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good_points.append(m)
                if len(kp_1) <= len(kp_2):
                    number_keypoints = len(kp_1)
                else:
                    number_keypoints = len(kp_2)
                car.append(title)
                try:
                    percentage_similarity.append(len(good_points) / number_keypoints * 100)
                except  :
                    print("An error has accured")
            Clock.schedule_once(partial(self.display_frame, original))
            cv2.waitKey(1)
            cv2.putText(original, (car[percentage_similarity.index(max(percentage_similarity))]+str(max(percentage_similarity))), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0), 1, 2)
            self.Similarity = max(percentage_similarity)
            self.Car = car[percentage_similarity.index(max(percentage_similarity))]
            # send this frame to the kivy Image Widget
            # Must use Clock.schedule_once to get this bit of code
            # to run back on the main thread (required for GUI operations)
            # the partial function just says to call the specified method with the provided argument (Clock adds a time argument)
        capture.release()
        cv2.destroyAllWindows()
    def display_frame(self, original, dt):
        # display the current video frame in the kivy Image widget
        # create a Texture the correct size and format for the frame
        texture = Texture.create(size=(original.shape[1], original.shape[0]), colorfmt='bgr')
        # copy the frame data into the texture
        texture.blit_buffer(original.tobytes(order=None), colorfmt='bgr', bufferfmt='ubyte')
        # flip the texture (otherwise the video is upside down
        texture.flip_vertical()
        # actually put the texture in the kivy Image widget
        self.main_screen.ids.vid.texture = texture
    def stop_vid(self):
        # stop the video capture loop
        self.do_vid = False
        print(self.Car)
        print(self.Similarity)
        f = open("CarModel.txt", "w")
        f.write(self.Car)

if __name__ == '__main__':
    MainApp().run()
