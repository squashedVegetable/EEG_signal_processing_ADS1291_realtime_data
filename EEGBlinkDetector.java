import ai.onnxruntime.*;
import org.jtransforms.fft.DoubleFFT_1D;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.File;
import java.util.List;
import java.util.Map;

/**
 * TODO: 
 * Maven:
 *
 * <dependency>
 *     <groupId>com.microsoft.onnxruntime</groupId>
 *     <artifactId>onnxruntime</artifactId>
 *     <version>1.17.1</version>
 * </dependency>
 *
 * <dependency>
 *     <groupId>com.github.wendykierp</groupId>
 *     <artifactId>JTransforms</artifactId>
 *     <version>3.1</version>
 * </dependency>
 *
 */
public class EEGBlinkDetector {
    private static final int SPS = 250;
    private static final double WINDOW_SECONDS = 0.8;
    private static final int WINDOW_SIZE = (int)(WINDOW_SECONDS * SPS);
    private double[] scalerMean;
    private double[] scalerStd;

    private void loadScaler(String scalerPath) throws IOException {
    ObjectMapper mapper = new ObjectMapper();
        Map<String, Object> data =mapper.readValue(new File(scalerPath), Map.class);
        List<Double> meanList = (List<Double>) data.get("mean");
        List<Double> scaleList = (List<Double>) data.get("scale");
        scalerMean = meanList.stream().mapToDouble(Double::doubleValue).toArray();
        scalerStd = scaleList.stream().mapToDouble(Double::doubleValue).toArray();
    }
    
    // ONNX
    private final OrtEnvironment env;
    private final OrtSession session;

    public EEGBlinkDetector(String onnxPath, String scalerPath) throws OrtException {
        env = OrtEnvironment.getEnvironment();
        session = env.createSession(onnxPath, new OrtSession.SessionOptions());
        
        loadScaler(scalerPath);
    }

    public int predict(double[] eegWindow) throws OrtException {
        // Filter
        double[] filtered = bandpassPlaceholder(eegWindow);
        // Feature extraction
        double[] features = extractFeatures(filtered);
        // Standardize features
        double[] scaled = scaleFeatures(features);
        // Convert to float tensor
        float[][] input = new float[1][scaled.length];
        for(int i = 0; i < scaled.length; i++) {
            input[0][i] = (float)scaled[i];
        }

        OnnxTensor tensor = OnnxTensor.createTensor(env, input);
        // TODO: Replace "float_input" with actual ONNX input name
        OrtSession.Result result = session.run(Collections.singletonMap("float_input", tensor));

        // TODO: Example output shape: [[0]] or [[1]]
        long[][] output = (long[][]) result.get(0).getValue();
        return (int)output[0][0];
    }

    private double[] extractFeatures(double[] window) {
        double mean = mean(window);
        double variance = variance(window, mean);
        double peakToPeak = peakToPeak(window);
        double skewness = skewness(window, mean);
        double kurtosis = kurtosis(window, mean);
        double deltaPower = bandPower(window, 0.5, 4.0);

        return new double[] {
                mean,
                variance,
                peakToPeak,
                skewness,
                kurtosis,
                deltaPower
        };
    }

    //Manual feauture calculation
    private double bandPower(double[] signal,double lowFreq, double highFreq) {
        int n = signal.length;
        double[] fftData = new double[n * 2];

        for(int i = 0; i < n; i++) {
            fftData[i] = signal[i];
        }
        DoubleFFT_1D fft = new DoubleFFT_1D(n);
        fft.realForwardFull(fftData);

        double power = 0.0;
        for(int k = 0; k < n / 2; k++) {
            double freq = (double)k * SPS / n;
            if(freq >= lowFreq && freq <= highFreq) {
                double real = fftData[2 * k];
                double imag = fftData[2 * k + 1];
                double magnitudeSquared = real * real + imag * imag;
                power += magnitudeSquared;
            }
        }
        return power;
    }

    private double[] scaleFeatures(double[] features) {
        double[] scaled = new double[features.length];

        for(int i = 0; i < features.length; i++) {
            scaled[i] =(features[i] - scalerMean[i])/scalerStd[i];
        }
        return scaled;
    }

    private double[] bandpassPlaceholder(double[] signal) {
        //TODO use bandpassfilter
        return signal;
    }

    private double mean(double[] x) {
        double sum = 0.0;
        for(double v : x) {
            sum += v;
        }
        return sum / x.length;
    }

    private double variance(double[] x, double mean) {
        double sum = 0.0;
        for(double v : x) {
            double d = v - mean;
            sum += d * d; 
        }
        return sum / x.length;
    }

    private double peakToPeak(double[] x) {
        double min = Double.MAX_VALUE;
        double max = -Double.MAX_VALUE;
        for(double v : x) {
            if(v < min) min = v;
            if(v > max) max = v;
        }
        return max - min;
    }

    private double skewness(double[] x, double mean) {
        double std = Math.sqrt(variance(x, mean));
        double sum = 0.0;

        for(double v : x) {
            sum += Math.pow((v - mean) / std, 3);
        }
        return sum / x.length;
    }

    private double kurtosis(double[] x, double mean) {
        double std = Math.sqrt(variance(x, mean));
        double sum = 0.0;
        for(double v : x) {
            sum += Math.pow((v - mean) / std, 4);
        }
        return sum / x.length;
    }
    
    public static void main(String[] args) throws Exception {
        EEEGBlinkDetector detector = new EEGBlinkDetector( "blink_model.onnx","scaler.json");
        // Example EEG window
        double[] eegWindow = new double[WINDOW_SIZE];
        //TODO: will window here
        for(int i = 0; i < eegWindow.length; i++) {
            eegWindow[i] = Math.sin(2 * Math.PI * 2 * i / SPS);
        }
        int prediction = detector.predict(eegWindow);
        System.out.println("Prediction: " + prediction);
        if(prediction == 1) { System.out.println("Blink detected"); }
    }
}
