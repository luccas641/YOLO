import torch
import argparse
import os

def convert_ckpt_to_pt(ckpt_path: str, output_path: str, device: str):
    """
    Cleans a .ckpt file by extracting and renaming EMA weights and converts it to a .pt file (weights only).
    """
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    print(checkpoint.keys())

    if "state_dict" not in checkpoint:
        print("Error: 'state_dict' not found in checkpoint. Cannot proceed with cleaning and conversion.")
        return

    state_dict = checkpoint["state_dict"]
    new_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("ema.model"):
            new_key = key.replace("ema.model", "model.model")
            new_state_dict[new_key] = value
    checkpoint["state_dict"] = new_state_dict
    # Save the cleaned state_dict as .pt file
    torch.save(checkpoint, output_path)
    print(f"✅ Cleaned and converted .ckpt to .pt: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean a .ckpt file and convert it to a .pt file.")
    parser.add_argument("ckpt_path", help="Path to the input .ckpt file.")
    parser.add_argument("output_path", help="Path for the final output .pt file.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for loading the checkpoint (e.g., 'cuda', 'cpu'). Defaults to 'cuda' if available, otherwise 'cpu'.")
    args = parser.parse_args()

    # Ensure paths are absolute for consistency, though relative paths work with argparse
    ckpt_path_abs = os.path.abspath(args.ckpt_path)
    output_path_abs = os.path.abspath(args.output_path)

    convert_ckpt_to_pt(ckpt_path_abs, output_path_abs, args.device)