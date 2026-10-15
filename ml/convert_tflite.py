import tensorflow as tf
import os

def convert_model(saved_model_dir, output_filename):
    if not os.path.exists(saved_model_dir):
        print(f"❌ Error: Directory '{saved_model_dir}' not found. Did you run onnx2tf?")
        return

    print(f"Converting {saved_model_dir} to {output_filename}...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    try:
        tflite_model = converter.convert()
        with open(output_filename, 'wb') as f:
            f.write(tflite_model)
        print(f"Successfully generated {output_filename}!\n")
    except Exception as e:
        print(f"Conversion failed for {output_filename}: {e}\n")

if __name__ == "__main__":
    print("=== Starting TFLite Conversions ===\n")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    
    # 1. Convert the Defense Model (Pre-game Fleet Placement)
    convert_model(os.path.join(data_dir, 'saved_model_defense'), os.path.join(data_dir, 'moriarty_defense.tflite'))
    
    # 2. Convert the Offense Model (Live Fire Prediction)
    convert_model(os.path.join(data_dir, 'saved_model_offense'), os.path.join(data_dir, 'moriarty_offense.tflite'))
    
    print("=== Conversion Complete ===")