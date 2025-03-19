import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import yaml
import time
from ultralytics import YOLO
from collections import Counter
import sys

class YOLOv8VideoDetectionPipeline:
    def __init__(self, model_path, conf_thresh=0.25, iou_thresh=0.45, img_size=640):
        """
        Initialize the YOLOv8 detection pipeline for videos and images.
        
        Args:
            model_path (str): Path to the fine-tuned YOLOv8 model weights
            conf_thresh (float): Confidence threshold for detections
            iou_thresh (float): IoU threshold for non-maximum suppression
            img_size (int): Image size for detection
        """
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.img_size = img_size
        
        # Load model
        print(f"Loading YOLOv8 model from {model_path}")
        self.model = YOLO(model_path)
        
        # Extract class names directly from the model
        self.class_names = self.model.names
        
        # If class names weren't loaded from the model, try to load from external files
        if not self.class_names:
            print("Class names not found in model. Trying to load from external files...")
            self.class_names = self._get_class_names(model_path)
            
        print(f"Loaded {len(self.class_names)} class names: {self.class_names}")
        
    def _get_class_names(self, model_path):
        """Try to extract class names from the model directory."""
        # First, try to find a data.yaml file in the parent directory
        model_dir = Path(model_path).parent
        yaml_path = model_dir / 'data.yaml'
        
        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                if 'names' in data:
                    return data['names']
        
        # If no YAML file found, try to look for classes.txt
        txt_path = model_dir / 'classes.txt'
        if txt_path.exists():
            with open(txt_path, 'r') as f:
                return [line.strip() for line in f.readlines()]
        
        # If no class names found, return a default dictionary
        print("Warning: Could not find class names. Objects will be labeled with their class index.")
        return {i: f"Class {i}" for i in range(100)}  # Default fallback names
    
    def process_frame(self, frame):
        """
        Process a single video frame with the YOLOv8 model.
        
        Args:
            frame: Video frame to process
            
        Returns:
            processed_frame: Frame with detections visualized
            detections: List of detections [class_name, confidence, x, y, width, height]
            class_counts: Dictionary of class counts
            avg_confidence: Average confidence of detections
        """
        # Perform detection
        results = self.model(frame, conf=self.conf_thresh, iou=self.iou_thresh, imgsz=self.img_size)
        
        # Process results
        detections = []
        class_counts = Counter()
        total_confidence = 0.0
        
        # Create a copy of the frame for visualization
        processed_frame = frame.copy()
        
        # Parse results
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get class ID and confidence
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                # Get class name
                class_name = self.class_names[cls_id]
                
                # Update class count
                class_counts[class_name] += 1
                
                # Add to total confidence
                total_confidence += conf
                
                # Convert to [class_name, confidence, x, y, width, height] format
                width = x2 - x1
                height = y2 - y1
                
                detections.append([class_name, conf, x1, y1, width, height])
                
                # Draw bounding box
                label = f"{class_name} {conf:.2f}"
                
                # Draw bounding box and label
                color = (0, 255, 0)  # Green color for bounding box
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with background
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(processed_frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
                cv2.putText(processed_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Calculate average confidence
        avg_confidence = total_confidence / len(detections) if detections else 0.0
        
        # Add class counts and average confidence to the frame
        y_offset = 30
        # Add detection quality info
        quality_text = f"Detection Quality: {self.get_accuracy_label(avg_confidence)}"
        quality_percentage = f"Est. Accuracy: {self.get_accuracy_percentage(avg_confidence):.1f}%"
        cv2.putText(processed_frame, quality_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += 30
        cv2.putText(processed_frame, quality_percentage, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += 30
        
        # Add class counts
        if class_counts:
            for class_name, count in class_counts.items():
                text = f"{class_name}: {count}"
                cv2.putText(processed_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                y_offset += 30
        
        return processed_frame, detections, dict(class_counts), avg_confidence
    
    def process_image(self, image_path, output_path=None, visualize=True):
        """
        Process a single image with the YOLOv8 model.
        
        Args:
            image_path (str): Path to the input image
            output_path (str, optional): Path to save the output image
            visualize (bool): Whether to visualize detections
            
        Returns:
            list: List of detections, each as [class_id, confidence, x, y, width, height]
            dict: Count of objects per class
            float: Average confidence of detections
        """
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image {image_path}")
            return [], {}, 0.0
        
        # Process the frame
        processed_frame, detections, class_counts, avg_confidence = self.process_frame(img)
        
        # Save the output image if output_path is provided
        if output_path and visualize:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, processed_frame)
        
        return detections, class_counts, avg_confidence
    
    def process_video(self, video_path, output_path=None, visualize=True, save_results=True, 
                      frame_interval=1, display_video=False, fps=None):
        """
        Process a video with the YOLOv8 model.
        
        Args:
            video_path (str): Path to the input video
            output_path (str, optional): Path to save the output video
            visualize (bool): Whether to visualize detections
            save_results (bool): Whether to save detection results to a text file
            frame_interval (int): Process every nth frame (1 = all frames)
            display_video (bool): Whether to display the video during processing
            fps (int, optional): FPS for the output video. If None, use input video's FPS
            
        Returns:
            dict: Frame-by-frame detections
            dict: Frame-by-frame class counts
            dict: Frame-by-frame confidence scores
            dict: Overall video statistics
        """
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return {}, {}, {}, {}
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        output_fps = fps if fps is not None else source_fps
        
        print(f"Video: {video_path}")
        print(f"Dimensions: {width}x{height} | FPS: {source_fps} | Total frames: {total_frames}")
        print(f"Processing every {frame_interval} frame(s)")
        
        # Prepare output video writer if needed
        video_writer = None
        if output_path and visualize:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'avc1' for .mp4 or 'XVID' for .avi
            video_writer = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
        
        # Initialize result containers
        frame_detections = {}
        frame_class_counts = {}
        frame_confidences = {}
        
        # Overall video statistics
        all_class_counts = Counter()
        total_confidence = 0.0
        processed_frames_count = 0
        
        # Process the video
        pbar = tqdm(total=total_frames, desc="Processing video")
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            pbar.update(1)
            
            # Process every nth frame
            if frame_idx % frame_interval == 0:
                # Process the frame
                processed_frame, detections, class_counts, avg_confidence = self.process_frame(frame)
                
                # Store results
                frame_detections[frame_idx] = detections
                frame_class_counts[frame_idx] = class_counts
                frame_confidences[frame_idx] = avg_confidence
                
                # Update overall statistics
                all_class_counts.update(class_counts)
                total_confidence += avg_confidence
                processed_frames_count += 1
                
                # Write to output video if requested
                if video_writer is not None:
                    video_writer.write(processed_frame)
                
                # Display the frame if requested
                if display_video:
                    cv2.imshow('YOLOv8 Detection', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                        break
            else:
                # Write the original frame to output video if requested
                if video_writer is not None:
                    video_writer.write(frame)
            
            frame_idx += 1
        
        # Calculate overall average confidence
        overall_avg_confidence = total_confidence / processed_frames_count if processed_frames_count > 0 else 0.0
        
        # Clean up
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if display_video:
            cv2.destroyAllWindows()
        pbar.close()
        
        # Create overall statistics
        video_stats = {
            'total_frames': total_frames,
            'processed_frames': processed_frames_count,
            'overall_avg_confidence': overall_avg_confidence,
            'estimated_accuracy': self.get_accuracy_percentage(overall_avg_confidence),
            'detection_quality': self.get_accuracy_label(overall_avg_confidence),
            'total_class_counts': dict(all_class_counts)
        }
        
        # Save results to a text file if requested
        if save_results and output_path:
            # Create a text file name based on the video output path
            results_path = os.path.splitext(output_path)[0] + '_results.txt'
            self.export_video_results(frame_detections, frame_confidences, frame_class_counts, video_stats, results_path)
        
        return frame_detections, frame_class_counts, frame_confidences, video_stats
    
    def export_video_results(self, frame_detections, frame_confidences, frame_class_counts, video_stats, output_path):
        """
        Export video detection results to a text file.
        
        Args:
            frame_detections (dict): Dictionary mapping frame indices to detection lists
            frame_confidences (dict): Dictionary mapping frame indices to confidence scores
            frame_class_counts (dict): Dictionary mapping frame indices to class counts
            video_stats (dict): Overall video statistics
            output_path (str): Path to save the results
        """
        with open(output_path, 'w') as f:
            f.write("===== VIDEO DETECTION SUMMARY =====\n\n")
            f.write(f"Total frames: {video_stats['total_frames']}\n")
            f.write(f"Processed frames: {video_stats['processed_frames']}\n")
            f.write(f"Overall Estimated Accuracy: {video_stats['estimated_accuracy']:.1f}%\n")
            f.write(f"Detection Quality: {video_stats['detection_quality']}\n\n")
            
            f.write("===== TOTAL OBJECT COUNTS ACROSS ALL FRAMES =====\n")
            for class_name, count in video_stats['total_class_counts'].items():
                f.write(f"{class_name}: {count}\n")
            
            f.write("\n===== DETAILED FRAME-BY-FRAME ANALYSIS =====\n")
            # Only show details for frames where objects were detected
            for frame_idx in sorted(frame_detections.keys()):
                detections = frame_detections[frame_idx]
                if not detections:
                    continue
                    
                f.write(f"\nFrame {frame_idx}:\n")
                avg_confidence = frame_confidences.get(frame_idx, 0.0)
                f.write(f"  Estimated Accuracy: {self.get_accuracy_percentage(avg_confidence):.1f}%\n")
                f.write(f"  Quality: {self.get_accuracy_label(avg_confidence)}\n")
                
                f.write("  Objects:\n")
                for class_name, count in frame_class_counts[frame_idx].items():
                    f.write(f"    {class_name}: {count}\n")
    
    def get_accuracy_percentage(self, avg_confidence):
        """
        Convert average confidence to an accuracy percentage.
        
        Args:
            avg_confidence (float): Average confidence of detections
            
        Returns:
            float: Estimated accuracy percentage
        """
        # Scale confidence to a more intuitive range (typically confidence >0.8 is very good)
        if avg_confidence > 0.8:
            # Scale 0.8-1.0 to 90-100%
            return 90 + (avg_confidence - 0.8) * 50
        elif avg_confidence > 0.6:
            # Scale 0.6-0.8 to 80-90%
            return 80 + (avg_confidence - 0.6) * 50
        elif avg_confidence > 0.4:
            # Scale 0.4-0.6 to 70-80%
            return 70 + (avg_confidence - 0.4) * 50
        else:
            # Scale 0.25-0.4 to 50-70%
            return max(50, 50 + (avg_confidence - 0.25) * 133)
    
    def get_accuracy_label(self, avg_confidence):
        """
        Convert average confidence to a human-readable label.
        
        Args:
            avg_confidence (float): Average confidence of detections
            
        Returns:
            str: Human-readable label for the quality of detections
        """
        if avg_confidence > 0.8:
            return "Excellent"
        elif avg_confidence > 0.6:
            return "Good"
        elif avg_confidence > 0.4:
            return "Moderate"
        else:
            return "Low"


def is_image_file(path):
    """Check if a file is an image based on its extension."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    return Path(path).suffix.lower() in image_extensions


def is_video_file(path):
    """Check if a file is a video based on its extension."""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']
    return Path(path).suffix.lower() in video_extensions


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Detection Pipeline for Videos and Images")
    parser.add_argument("--model", required=True, help="Path to the YOLOv8 model weights (.pt file)")
    parser.add_argument("--input", required=True, nargs='+', help="Path to input directory or video/image file(s)")
    parser.add_argument("--output", default="output", help="Path to output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--no-visualize", action="store_true", help="Don't visualize detections")
    parser.add_argument("--no-save-results", action="store_true", help="Don't save detection results to text files")
    parser.add_argument("--frame-interval", type=int, default=1, help="Process every nth frame in videos")
    parser.add_argument("--display", action="store_true", help="Display videos while processing")
    parser.add_argument("--fps", type=float, help="FPS for output videos (default: same as input)")
    
    args = parser.parse_args()
    
    # Create the pipeline
    pipeline = YOLOv8VideoDetectionPipeline(
        model_path=args.model,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        img_size=args.img_size
    )
    
    # Process the input(s)
    start_time = time.time()
    
    # Categorize inputs into directories, videos, and images
    input_paths = args.input
    dir_paths = []
    video_paths = []
    image_paths = []
    
    for input_path in input_paths:
        path = Path(input_path)
        if path.is_dir():
            dir_paths.append(path)
        elif is_video_file(path):
            video_paths.append(path)
        elif is_image_file(path):
            image_paths.append(path)
        else:
            print(f"Warning: {input_path} is not a valid directory, video, or image file. Skipping.")
    
    # Process directories - collect all files
    for dir_path in dir_paths:
        print(f"Scanning directory: {dir_path}")
        # Find all video files
        for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']:
            video_paths.extend(list(dir_path.glob(f'**/*{ext}')))
        # Find all image files
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            image_paths.extend(list(dir_path.glob(f'**/*{ext}')))
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Process videos
    video_results = {}
    for video_path in video_paths:
        print(f"\nProcessing video: {video_path}")
        output_path = os.path.join(args.output, Path(video_path).name)
        frame_detections, frame_class_counts, frame_confidences, video_stats = pipeline.process_video(
            video_path=str(video_path),
            output_path=output_path,
            visualize=not args.no_visualize,
            save_results=not args.no_save_results,
            frame_interval=args.frame_interval,
            display_video=args.display,
            fps=args.fps
        )
        
        video_results[str(video_path)] = video_stats
        
        # Print video summary
        print(f"Video {video_path.name} processed:")
        print(f"  - Processed {video_stats['processed_frames']} of {video_stats['total_frames']} frames")
        print(f"  - Estimated Accuracy: {video_stats['estimated_accuracy']:.1f}%")
        print(f"  - Detection Quality: {video_stats['detection_quality']}")
        
        # Print class counts
        if video_stats['total_class_counts']:
            print("  - Objects detected:")
            for class_name, count in video_stats['total_class_counts'].items():
                print(f"    * {class_name}: {count}")
        else:
            print("  - No objects detected")
    
    # Process images
    image_results = []
    if image_paths:
        print(f"\nProcessing {len(image_paths)} images")
        for img_path in tqdm(image_paths, desc="Processing images"):
            output_path = os.path.join(args.output, img_path.name)
            detections, class_counts, avg_confidence = pipeline.process_image(
                image_path=str(img_path),
                output_path=output_path,
                visualize=not args.no_visualize
            )
            
            image_results.append({
                'path': str(img_path),
                'detections': len(detections),
                'class_counts': class_counts,
                'avg_confidence': avg_confidence,
                'accuracy': pipeline.get_accuracy_percentage(avg_confidence),
                'quality': pipeline.get_accuracy_label(avg_confidence)
            })
    
    elapsed_time = time.time() - start_time
    
    # Print summary
    print(f"\n===== PROCESSING SUMMARY =====")
    print(f"Processed {len(video_paths)} videos and {len(image_paths)} images in {elapsed_time:.2f} seconds")
    
    if video_paths:
        print(f"\nVideo processing results:")
        for video_path, stats in video_results.items():
            print(f"  {Path(video_path).name}:")
            print(f"    - Estimated Accuracy: {stats['estimated_accuracy']:.1f}%")
            print(f"    - Objects detected: {sum(stats['total_class_counts'].values())}")
    
    if image_paths:
        print(f"\nImage processing summary:")
        total_objects = sum(sum(result['class_counts'].values()) for result in image_results)
        avg_accuracy = sum(result['accuracy'] for result in image_results) / len(image_results) if image_results else 0
        print(f"  - Total objects detected: {total_objects}")
        print(f"  - Average estimated accuracy: {avg_accuracy:.1f}%")
    
    print(f"\nOutput saved to {args.output}")


if __name__ == "__main__":
    main()