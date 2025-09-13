import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

class CRNNWithCTC:
    def __init__(self, img_width=256, img_height=64, max_length=20):
        self.img_width = img_width
        self.img_height = img_height
        self.max_length = max_length
        
        # Character set (can be customized)
        characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ '
        self.char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)
        self.num_to_char = layers.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True
        )
        
        self.vocab_size = len(characters) + 1  # +1 for CTC blank token
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Characters: {characters}")
    
    def create_model(self):
        """
        Create CRNN model with detailed architecture explanation
        """
        # INPUT LAYER
        input_img = layers.Input(shape=(self.img_height, self.img_width, 1), name="image")
        print(f"\n=== ARCHITECTURE BREAKDOWN ===")
        print(f"Input shape: {(self.img_height, self.img_width, 1)}")
        
        # =================== CNN COMPONENT ===================
        print(f"\n--- CNN FEATURE EXTRACTION ---")
        
        # Block 1: Initial feature detection
        x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv1")(input_img)
        x = layers.BatchNormalization(name="bn1")(x)
        x = layers.MaxPooling2D((2, 2), name="pool1")(x)  # 64x256 -> 32x128
        print(f"After Block 1: {x.shape} - Detects basic edges and strokes")
        
        # Block 2: Edge combination  
        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2")(x)
        x = layers.BatchNormalization(name="bn2")(x)
        x = layers.MaxPooling2D((2, 2), name="pool2")(x)  # 32x128 -> 16x64
        print(f"After Block 2: {x.shape} - Combines edges into character parts")
        
        # Block 3: Character part detection
        x = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv3")(x)
        x = layers.BatchNormalization(name="bn3")(x)
        x = layers.MaxPooling2D((2, 1), name="pool3")(x)  # 16x64 -> 8x64 (preserve width)
        print(f"After Block 3: {x.shape} - Detects character-like features")
        
        # Block 4: High-level features
        x = layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="conv4")(x)
        x = layers.BatchNormalization(name="bn4")(x)
        x = layers.MaxPooling2D((2, 1), name="pool4")(x)  # 8x64 -> 4x64
        print(f"After Block 4: {x.shape} - Complex character representations")
        
        # Block 5: Final feature extraction
        x = layers.Conv2D(512, (3, 3), activation="relu", padding="same", name="conv5")(x)
        x = layers.BatchNormalization(name="bn5")(x)
        x = layers.MaxPooling2D((2, 1), name="pool5")(x)  # 4x64 -> 2x64
        print(f"After Block 5: {x.shape} - Rich feature representations")
        
        # =================== SEQUENCE PREPARATION ===================
        print(f"\n--- SEQUENCE PREPARATION ---")
        
        # Reshape CNN output for RNN input
        # From (batch, height, width, channels) to (batch, width, height*channels)
        new_shape = ((x.shape[-2] * x.shape[-1]), x.shape[-3])  # (2*512, 64) = (1024, 64)
        x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
        print(f"After Reshape: {x.shape} - Convert to sequence format")
        print(f"  • {x.shape[-1]} timesteps (width dimension)")
        print(f"  • {x.shape[-2]} features per timestep (height × channels)")
        
        # Dense layer to adjust feature dimension
        x = layers.Dense(256, activation="relu", name="dense_features")(x)
        x = layers.Dropout(0.2)(x)
        print(f"After Dense: {x.shape} - Optimized feature dimension")
        
        # =================== RNN COMPONENT ===================
        print(f"\n--- RNN SEQUENCE MODELING ---")
        
        # Bidirectional LSTM layers
        x = layers.Bidirectional(
            layers.LSTM(256, return_sequences=True, dropout=0.2), 
            name="bidirectional_lstm_1"
        )(x)
        print(f"After BiLSTM 1: {x.shape} - Captures left-right context")
        
        x = layers.Bidirectional(
            layers.LSTM(256, return_sequences=True, dropout=0.2),
            name="bidirectional_lstm_2"
        )(x)
        print(f"After BiLSTM 2: {x.shape} - Deep sequence understanding")
        
        # =================== CTC OUTPUT LAYER ===================
        print(f"\n--- CTC OUTPUT LAYER ---")
        
        # Dense layer for character classification (including CTC blank)
        x = layers.Dense(self.vocab_size, activation="softmax", name="ctc_output")(x)
        print(f"Final Output: {x.shape} - Character probabilities per timestep")
        print(f"  • {x.shape[-1]} classes (characters + CTC blank)")
        print(f"  • {x.shape[-2]} timesteps for CTC alignment")
        
        # Create model
        model = keras.Model(inputs=input_img, outputs=x, name="CRNN_CTC")
        return model
    
    def ctc_loss_function(self, y_true, y_pred):
        """
        CTC loss function with detailed explanation
        """
        # y_true: ground truth labels (sparse format)
        # y_pred: predicted character probabilities per timestep
        
        # Get sequence lengths
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
        
        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        
        # CTC loss calculation
        loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
        return loss
    
    def demonstrate_ctc_alignment(self):
        """
        Demonstrate how CTC handles alignment with examples
        """
        print("\n=== CTC ALIGNMENT DEMONSTRATION ===")
        
        # Example: Target text "HELLO"
        target_text = "HELLO"
        sequence_length = 10  # RNN output length
        
        print(f"Target text: '{target_text}'")
        print(f"RNN output length: {sequence_length} timesteps")
        print(f"\nPossible CTC alignments:")
        
        # Show different valid alignments
        alignments = [
            "HEL-LO----",
            "-HE-L-LO--", 
            "H-E-L-L-O-",
            "HH-ELL-OO-",
            "--HELLO---"
        ]
        
        for i, alignment in enumerate(alignments, 1):
            print(f"  {i}. [{']['.join(list(alignment))}]")
            # Simulate CTC decoding
            decoded = self.decode_ctc_alignment(alignment)
            print(f"     → Decodes to: '{decoded}'")
        
        print(f"\nKey CTC Rules:")
        print(f"  1. '-' represents CTC blank (no character)")
        print(f"  2. Consecutive identical characters are merged")
        print(f"  3. Characters separated by blank are kept separate")
        print(f"  4. All valid alignments contribute to training")
    
    def decode_ctc_alignment(self, alignment):
        """
        Simulate CTC decoding process
        """
        decoded = ""
        prev_char = None
        
        for char in alignment:
            if char == '-':  # CTC blank
                prev_char = None
            elif char != prev_char:  # New character or after blank
                decoded += char
                prev_char = char
            # Skip if same as previous (merge consecutive)
        
        return decoded
    
    def visualize_feature_flow(self, sample_image):
        """
        Visualize how features flow through the network
        """
        model = self.create_model()
        
        print("\n=== FEATURE FLOW VISUALIZATION ===")
        
        # Create intermediate models to extract features at each stage
        cnn_output = keras.Model(inputs=model.input, outputs=model.get_layer("pool5").output)
        reshape_output = keras.Model(inputs=model.input, outputs=model.get_layer("reshape").output)
        lstm_output = keras.Model(inputs=model.input, outputs=model.get_layer("bidirectional_lstm_2").output)
        
        # Process sample image
        sample_batch = np.expand_dims(sample_image, axis=0)
        
        # Extract features at each stage
        cnn_features = cnn_output.predict(sample_batch, verbose=0)
        sequence_features = reshape_output.predict(sample_batch, verbose=0)
        lstm_features = lstm_output.predict(sample_batch, verbose=0)
        final_output = model.predict(sample_batch, verbose=0)
        
        print(f"Original Image: {sample_image.shape}")
        print(f"CNN Features: {cnn_features.shape}")
        print(f"Sequence Features: {sequence_features.shape}") 
        print(f"LSTM Features: {lstm_features.shape}")
        print(f"Final Output: {final_output.shape}")
        
        # Show character probabilities for first few timesteps
        print(f"\nCharacter probabilities for first 5 timesteps:")
        characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ -'
        
        for t in range(min(5, final_output.shape[1])):
            probs = final_output[0, t, :]
            top_char_idx = np.argmax(probs)
            top_char = characters[top_char_idx] if top_char_idx < len(characters) else 'BLANK'
            top_prob = probs[top_char_idx]
            
            print(f"  Timestep {t}: '{top_char}' (prob: {top_prob:.3f})")
    
    def training_example(self):
        """
        Show complete training example
        """
        print("\n=== TRAINING PROCESS ===")
        
        # Create model
        model = self.create_model()
        
        # Compile with CTC loss
        model.compile(optimizer='adam', loss=self.ctc_loss_function)
        
        print(f"\nModel compiled with:")
        print(f"  • Optimizer: Adam")
        print(f"  • Loss: CTC Loss")
        print(f"  • Total parameters: {model.count_params():,}")
        
        # Show training data format
        print(f"\nTraining Data Format:")
        print(f"  • Images: (batch_size, {self.img_height}, {self.img_width}, 1)")
        print(f"  • Labels: (batch_size, max_label_length) - sparse encoded")
        print(f"  • CTC handles variable length automatically")
        
        return model
    
    def inference_example(self, model, test_image):
        """
        Show complete inference process
        """
        print("\n=== INFERENCE PROCESS ===")
        
        # Predict
        batch_image = np.expand_dims(test_image, axis=0)
        predictions = model.predict(batch_image, verbose=0)
        
        print(f"Input image shape: {test_image.shape}")
        print(f"Raw predictions shape: {predictions.shape}")
        
        # CTC Decoding - Greedy approach
        input_length = np.array([predictions.shape[1]])
        
        # Use TensorFlow's CTC decoder
        decoded, _ = tf.keras.backend.ctc_decode(
            predictions, input_length, greedy=True
        )
        
        # Convert to text
        decoded_text = self.num_to_char(decoded[0]).numpy().decode('utf-8')
        
        print(f"Decoded text: '{decoded_text}'")
        
        return decoded_text

# =================== DEMONSTRATION ===================

def main_demonstration():
    """
    Run complete CRNN+CTC demonstration
    """
    print("🔥 CRNN + CTC Architecture Deep Dive 🔥")
    print("=" * 50)
    
    # Initialize CRNN+CTC system
    crnn_ctc = CRNNWithCTC(img_width=256, img_height=64)
    
    # 1. Show architecture breakdown
    model = crnn_ctc.create_model()
    
    # 2. Demonstrate CTC alignment
    crnn_ctc.demonstrate_ctc_alignment()
    
    # 3. Create sample image (simulated)
    sample_image = np.random.random((64, 256, 1)) * 255
    sample_image = sample_image.astype(np.float32) / 255.0
    
    # 4. Visualize feature flow
    crnn_ctc.visualize_feature_flow(sample_image)
    
    # 5. Show training setup
    trained_model = crnn_ctc.training_example()
    
    # 6. Show inference process
    crnn_ctc.inference_example(trained_model, sample_image)
    
    print(f"\n🎯 KEY TAKEAWAYS:")
    print(f"  • CNN extracts spatial features from text images")
    print(f"  • RNN models character sequences and context") 
    print(f"  • CTC handles alignment without character segmentation")
    print(f"  • End-to-end training optimizes all components together")
    print(f"  • Result: Robust text recognition from images")

if __name__ == "__main__":
    main_demonstration()

# =================== DETAILED MATHEMATICAL BREAKDOWN ===================

def mathematical_breakdown():
    """
    Show the mathematical operations in detail
    """
    print("\n" + "="*60)
    print("MATHEMATICAL BREAKDOWN")
    print("="*60)
    
    print("\n1. CNN CONVOLUTION OPERATION:")
    print("   Feature[i,j] = Σ(k,l) Input[i-k:i+k, j-l:j+l] ⊙ Kernel[k,l] + Bias")
    print("   Where ⊙ represents element-wise multiplication")
    
    print("\n2. MAXPOOLING OPERATION:")
    print("   Output[i,j] = max(Input[i*stride:(i+1)*stride, j*stride:(j+1)*stride])")
    
    print("\n3. LSTM CELL COMPUTATIONS:")
    print("   f_t = σ(W_f·[h_{t-1}, x_t] + b_f)  # Forget gate")
    print("   i_t = σ(W_i·[h_{t-1}, x_t] + b_i)  # Input gate") 
    print("   C̃_t = tanh(W_c·[h_{t-1}, x_t] + b_c)  # Candidate")
    print("   C_t = f_t * C_{t-1} + i_t * C̃_t  # Cell state")
    print("   o_t = σ(W_o·[h_{t-1}, x_t] + b_o)  # Output gate")
    print("   h_t = o_t * tanh(C_t)  # Hidden state")
    
    print("\n4. CTC LOSS COMPUTATION:")
    print("   L = -log(Σ_π P(π|x))  # Sum over all valid alignments π")
    print("   Where P(π|x) = Π_t y_{π_t,t}  # Product of character probabilities")
    
    print("\n5. CTC FORWARD ALGORITHM:")
    print("   α_t(s) = Probability of partial alignment up to time t, state s")
    print("   α_t(s) = (α_{t-1}(s) + α_{t-1}(s-1)) * y_{l_s,t}")
    print("   Where l_s is the label at state s")
    
    print("\n6. BIDIRECTIONAL LSTM:")
    print("   h⃗_t = LSTM_forward(x_t, h⃗_{t-1})")
    print("   h⃖_t = LSTM_backward(x_t, h⃖_{t+1})")  
    print("   h_t = [h⃗_t; h⃖_t]  # Concatenation")

# Run the demonstration
if __name__ == "__main__":
    main_demonstration()
    mathematical_breakdown()