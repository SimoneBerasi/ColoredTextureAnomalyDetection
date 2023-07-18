# ColoredTextureAnomalyDetection

To utilize the code in this repository, please follow the steps outlined below:
1. Clone the repository to your local machine using the following command:

 ``` git clone  https://github.com/SimoneBerasi/CCW-SSIMautoencoder-for-anomaly-detection.git ```

2. Navigate to the repository’s root directory:

```cd CW-SSIM-autoencoder-for-anomaly-detection```

3. Ensure that you have Python 3 installed on your machine, along with the required
dependencies. You can install the necessary dependencies using the following command:

```pip install -r requirements.txt```

4. Configure the parameters for the project by editing the parameters.ini file located
in the Config folder. Adjust the values according to your specific requirements.

5. Run the code by executing the Main.py file with the appropriate arguments. Use
the following command to start the desired phase:

```python Main.py -p [phase]```

Replace [phase] with one of the following values:

• training: Starts the training phase.

• prediction: Starts the prediction phase.

• evaluation: Starts the evaluation phase.
