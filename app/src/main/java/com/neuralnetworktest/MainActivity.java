package com.neuralnetworktest;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.text.method.ScrollingMovementMethod;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.neuralnetworktest.ml.Model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    Button camera, gallery;
    ImageView imageView;
    TextView result;
    int imageSize = 224;

    private void predictImage(Bitmap image) {
        try {
            Model model = Model.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            int currentPixel = 0;
            //iterate over each pixel and extract R, G, and B values. Add those values individually to the byte buffer.
            for (int i = 0; i < imageSize; i++) {
                for (int j = 0; j < imageSize; j++) {
                    int val = intValues[currentPixel++]; //take value and increment currentPixel //RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255));

                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            // find the index of the class with the biggest confidence.
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }

            String[] classes = {"apple", "banana",  "beetroot",  "bell pepper",  "cabbage", "capsicum", "carrot", "cauliflower", "cheesecake",
                    "chilli pepper", "corn", "cucumber", "cup cakes", "eggplant", "fried rice", "garlic", "ginger", "grapes", "jalepeno",
                    "kiwi", "lemon", "lettuce", "mango", "onion", "orange", "paprika", "pear", "peas", "pineapple", "pomegranate", "potato",
                    "raddish", "soy beans", "spinach", "sweetcorn", "sweetpotato", "tomato", "turnip", "watermelon"};

            String results = "This is ";
            results+= classes[maxPos] + "\n";

            //List of classes with possibilities
            for (int i = 0; i < confidences.length; i++) {
                results+= classes[i] + ": " + confidences[i] +"\n";
            }
            //result.setText(classes[maxPos]);
            result.setText(results);
            result.setMovementMethod(new ScrollingMovementMethod());

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (resultCode == RESULT_OK){
            if (requestCode == 3){ //Picture from camera
                Bitmap image = (Bitmap) data.getExtras().get("data"); //get image

                //resize image to square
                int dimension = Math.min(image.getWidth(), image.getHeight());
                image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);

                //display image to ImageView
                imageView.setImageBitmap(image);

                //Scale image to 224x224
                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                //Predict
                predictImage(image);
            } else { //Picture from gallery
                Uri dataUri = data.getData();
                //Get image from gallery
                Bitmap image = null;
                try {
                    image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dataUri);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                //display image to ImageView
                imageView.setImageBitmap(image);
                //Scale image to 224x224
                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                //Predict
                predictImage(image);
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        camera = findViewById(R.id.button);
        gallery = findViewById(R.id.button2);

        result = findViewById(R.id.result);
        imageView = findViewById(R.id.imageView);

        camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                    if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED){
                        Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                        startActivityForResult(cameraIntent, 3);
                    } else {
                        requestPermissions(new String[] {Manifest.permission.CAMERA}, 100);
                    }
                }
            }
        });
        gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent galleryIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(galleryIntent, 1);
            }
        });
    }

}