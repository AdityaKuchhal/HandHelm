#!/usr/bin/env python3
"""
Computer Control Actions for Hand Helm

This module implements various computer control actions that can be triggered
by hand gestures, including media control, presentation navigation, cursor control,
and system actions.
"""

import os
import sys
import time
import platform
from typing import Dict, Callable, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DEFAULT_GESTURE_ACTIONS

try:
    import pyautogui
    import keyboard
    from pynput import mouse, keyboard as pynput_keyboard
    CONTROL_AVAILABLE = True
except ImportError:
    CONTROL_AVAILABLE = False
    print("⚠️ Control libraries not available. Install pyautogui, keyboard, and pynput for full functionality.")

class ComputerController:
    """
    A comprehensive computer control system for gesture-based interactions.
    
    This class provides:
    - Media playback control (play, pause, volume, skip)
    - Presentation navigation (next/previous slide)
    - Cursor and mouse control
    - Keyboard shortcuts and system actions
    - Customizable action mappings
    """
    
    def __init__(self, gesture_actions: Optional[Dict[str, str]] = None):
        """
        Initialize the computer controller.
        
        Args:
            gesture_actions (dict): Custom gesture to action mappings
        """
        self.gesture_actions = gesture_actions or DEFAULT_GESTURE_ACTIONS.copy()
        self.action_functions = self._initialize_action_functions()
        self.last_action_time = {}
        self.action_cooldown = 0.5  # Minimum time between actions (seconds)
        
        # Disable pyautogui failsafe for smoother operation
        if CONTROL_AVAILABLE:
            pyautogui.FAILSAFE = False
            pyautogui.PAUSE = 0.1
    
    def _initialize_action_functions(self) -> Dict[str, Callable]:
        """Initialize all available action functions."""
        return {
            # Media Control
            'play_media': self._play_media,
            'pause_media': self._pause_media,
            'stop_media': self._stop_media,
            'next_track': self._next_track,
            'previous_track': self._previous_track,
            'volume_up': self._volume_up,
            'volume_down': self._volume_down,
            'mute_volume': self._mute_volume,
            
            # Presentation Control
            'next_slide': self._next_slide,
            'previous_slide': self._previous_slide,
            'start_presentation': self._start_presentation,
            'end_presentation': self._end_presentation,
            
            # Cursor Control
            'scroll_up': self._scroll_up,
            'scroll_down': self._scroll_down,
            'scroll_left': self._scroll_left,
            'scroll_right': self._scroll_right,
            'left_click': self._left_click,
            'right_click': self._right_click,
            'double_click': self._double_click,
            
            # Browser Control
            'next_tab': self._next_tab,
            'previous_tab': self._previous_tab,
            'new_tab': self._new_tab,
            'close_tab': self._close_tab,
            'refresh_page': self._refresh_page,
            'go_back': self._go_back,
            'go_forward': self._go_forward,
            
            # System Control
            'alt_tab': self._alt_tab,
            'alt_shift_tab': self._alt_shift_tab,
            'minimize_window': self._minimize_window,
            'maximize_window': self._maximize_window,
            'close_window': self._close_window,
            'screenshot': self._take_screenshot,
            
            # Custom Actions
            'custom_action': self._custom_action,
            'no_action': self._no_action
        }
    
    def execute_action(self, gesture: str) -> bool:
        """
        Execute the action associated with a gesture.
        
        Args:
            gesture (str): The detected gesture
        
        Returns:
            bool: True if action was executed successfully
        """
        if not CONTROL_AVAILABLE:
            print(f"⚠️ Control not available for gesture: {gesture}")
            return False
        
        # Check cooldown
        current_time = time.time()
        if gesture in self.last_action_time:
            if current_time - self.last_action_time[gesture] < self.action_cooldown:
                return False
        
        # Get action
        action = self.gesture_actions.get(gesture, 'no_action')
        
        # Execute action
        if action in self.action_functions:
            try:
                self.action_functions[action]()
                self.last_action_time[gesture] = current_time
                print(f"✅ Executed action: {action} for gesture: {gesture}")
                return True
            except Exception as e:
                print(f"❌ Error executing action {action}: {e}")
                return False
        else:
            print(f"⚠️ Unknown action: {action}")
            return False
    
    def set_gesture_action(self, gesture: str, action: str) -> bool:
        """
        Set a custom gesture to action mapping.
        
        Args:
            gesture (str): The gesture name
            action (str): The action to execute
        
        Returns:
            bool: True if mapping was set successfully
        """
        if action in self.action_functions:
            self.gesture_actions[gesture] = action
            print(f"✅ Mapped gesture '{gesture}' to action '{action}'")
            return True
        else:
            print(f"❌ Unknown action: {action}")
            return False
    
    def get_available_actions(self) -> list:
        """Get list of all available actions."""
        return list(self.action_functions.keys())
    
    def get_gesture_mappings(self) -> Dict[str, str]:
        """Get current gesture to action mappings."""
        return self.gesture_actions.copy()
    
    # Media Control Actions
    def _play_media(self):
        """Play media (spacebar)."""
        pyautogui.press('space')
    
    def _pause_media(self):
        """Pause media (spacebar)."""
        pyautogui.press('space')
    
    def _stop_media(self):
        """Stop media (escape)."""
        pyautogui.press('escape')
    
    def _next_track(self):
        """Next track (right arrow)."""
        pyautogui.press('right')
    
    def _previous_track(self):
        """Previous track (left arrow)."""
        pyautogui.press('left')
    
    def _volume_up(self):
        """Increase volume (up arrow)."""
        pyautogui.press('up')
    
    def _volume_down(self):
        """Decrease volume (down arrow)."""
        pyautogui.press('down')
    
    def _mute_volume(self):
        """Mute/unmute volume (m key)."""
        pyautogui.press('m')
    
    # Presentation Control Actions
    def _next_slide(self):
        """Next slide (right arrow or page down)."""
        pyautogui.press('right')
    
    def _previous_slide(self):
        """Previous slide (left arrow or page up)."""
        pyautogui.press('left')
    
    def _start_presentation(self):
        """Start presentation (F5)."""
        pyautogui.press('f5')
    
    def _end_presentation(self):
        """End presentation (escape)."""
        pyautogui.press('escape')
    
    # Cursor Control Actions
    def _scroll_up(self):
        """Scroll up."""
        pyautogui.scroll(3)
    
    def _scroll_down(self):
        """Scroll down."""
        pyautogui.scroll(-3)
    
    def _scroll_left(self):
        """Scroll left (shift + scroll)."""
        pyautogui.hotkey('shift', 'scroll', 3)
    
    def _scroll_right(self):
        """Scroll right (shift + scroll)."""
        pyautogui.hotkey('shift', 'scroll', -3)
    
    def _left_click(self):
        """Left mouse click."""
        pyautogui.click()
    
    def _right_click(self):
        """Right mouse click."""
        pyautogui.rightClick()
    
    def _double_click(self):
        """Double click."""
        pyautogui.doubleClick()
    
    # Browser Control Actions
    def _next_tab(self):
        """Next tab (Ctrl + Tab)."""
        pyautogui.hotkey('ctrl', 'tab')
    
    def _previous_tab(self):
        """Previous tab (Ctrl + Shift + Tab)."""
        pyautogui.hotkey('ctrl', 'shift', 'tab')
    
    def _new_tab(self):
        """New tab (Ctrl + T)."""
        pyautogui.hotkey('ctrl', 't')
    
    def _close_tab(self):
        """Close tab (Ctrl + W)."""
        pyautogui.hotkey('ctrl', 'w')
    
    def _refresh_page(self):
        """Refresh page (F5)."""
        pyautogui.press('f5')
    
    def _go_back(self):
        """Go back (Alt + Left)."""
        pyautogui.hotkey('alt', 'left')
    
    def _go_forward(self):
        """Go forward (Alt + Right)."""
        pyautogui.hotkey('alt', 'right')
    
    # System Control Actions
    def _alt_tab(self):
        """Alt + Tab (switch windows)."""
        pyautogui.hotkey('alt', 'tab')
    
    def _alt_shift_tab(self):
        """Alt + Shift + Tab (switch windows backwards)."""
        pyautogui.hotkey('alt', 'shift', 'tab')
    
    def _minimize_window(self):
        """Minimize window (Cmd + M on Mac, Win + Down on Windows)."""
        if platform.system() == 'Darwin':  # macOS
            pyautogui.hotkey('cmd', 'm')
        else:  # Windows/Linux
            pyautogui.hotkey('win', 'down')
    
    def _maximize_window(self):
        """Maximize window (Cmd + Ctrl + F on Mac, Win + Up on Windows)."""
        if platform.system() == 'Darwin':  # macOS
            pyautogui.hotkey('cmd', 'ctrl', 'f')
        else:  # Windows/Linux
            pyautogui.hotkey('win', 'up')
    
    def _close_window(self):
        """Close window (Cmd + W on Mac, Alt + F4 on Windows)."""
        if platform.system() == 'Darwin':  # macOS
            pyautogui.hotkey('cmd', 'w')
        else:  # Windows/Linux
            pyautogui.hotkey('alt', 'f4')
    
    def _take_screenshot(self):
        """Take screenshot (Cmd + Shift + 3 on Mac, Win + Shift + S on Windows)."""
        if platform.system() == 'Darwin':  # macOS
            pyautogui.hotkey('cmd', 'shift', '3')
        else:  # Windows/Linux
            pyautogui.hotkey('win', 'shift', 's')
    
    # Custom Actions
    def _custom_action(self):
        """Custom action - can be overridden."""
        print("🎯 Custom action executed!")
    
    def _no_action(self):
        """No action - placeholder."""
        pass

class GestureActionManager:
    """
    A higher-level manager for gesture-based actions with additional features.
    
    This class provides:
    - Action history and logging
    - Gesture sequence recognition
    - Action confirmation and safety checks
    - Performance monitoring
    """
    
    def __init__(self, controller: ComputerController):
        """
        Initialize the gesture action manager.
        
        Args:
            controller (ComputerController): The computer controller instance
        """
        self.controller = controller
        self.action_history = []
        self.gesture_sequence = []
        self.sequence_timeout = 2.0  # seconds
        self.max_history = 100
    
    def process_gesture(self, gesture: str, confidence: float = 1.0) -> bool:
        """
        Process a detected gesture and execute appropriate action.
        
        Args:
            gesture (str): The detected gesture
            confidence (float): Confidence level (0.0 to 1.0)
        
        Returns:
            bool: True if action was executed
        """
        current_time = time.time()
        
        # Add to gesture sequence
        self.gesture_sequence.append({
            'gesture': gesture,
            'confidence': confidence,
            'timestamp': current_time
        })
        
        # Clean old gestures from sequence
        self.gesture_sequence = [
            g for g in self.gesture_sequence 
            if current_time - g['timestamp'] < self.sequence_timeout
        ]
        
        # Execute action
        success = self.controller.execute_action(gesture)
        
        # Log action
        if success:
            self.action_history.append({
                'gesture': gesture,
                'confidence': confidence,
                'action': self.controller.gesture_actions.get(gesture, 'no_action'),
                'timestamp': current_time,
                'success': True
            })
            
            # Limit history size
            if len(self.action_history) > self.max_history:
                self.action_history = self.action_history[-self.max_history:]
        
        return success
    
    def get_action_history(self, limit: int = 10) -> list:
        """Get recent action history."""
        return self.action_history[-limit:] if limit else self.action_history
    
    def get_gesture_sequence(self) -> list:
        """Get current gesture sequence."""
        return self.gesture_sequence.copy()
    
    def clear_history(self):
        """Clear action history."""
        self.action_history.clear()
        self.gesture_sequence.clear()
    
    def get_statistics(self) -> dict:
        """Get action statistics."""
        if not self.action_history:
            return {'total_actions': 0, 'success_rate': 0.0}
        
        total_actions = len(self.action_history)
        successful_actions = sum(1 for action in self.action_history if action['success'])
        success_rate = successful_actions / total_actions
        
        # Gesture frequency
        gesture_counts = {}
        for action in self.action_history:
            gesture = action['gesture']
            gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
        
        return {
            'total_actions': total_actions,
            'successful_actions': successful_actions,
            'success_rate': success_rate,
            'gesture_frequency': gesture_counts
        }

def main():
    """Main function for testing computer control."""
    print("🎮 Hand Helm - Computer Control Test")
    print("=" * 40)
    
    if not CONTROL_AVAILABLE:
        print("❌ Control libraries not available.")
        print("Install required packages:")
        print("  pip install pyautogui keyboard pynput")
        return
    
    # Initialize controller
    controller = ComputerController()
    manager = GestureActionManager(controller)
    
    print("✅ Computer controller initialized!")
    print(f"📋 Available actions: {len(controller.get_available_actions())}")
    print(f"🎯 Gesture mappings: {len(controller.get_gesture_mappings())}")
    
    # Test some actions
    print("\n🧪 Testing actions...")
    
    test_gestures = ['fist', 'open_palm', 'thumbs_up', 'thumbs_down']
    
    for gesture in test_gestures:
        print(f"\nTesting gesture: {gesture}")
        success = manager.process_gesture(gesture, confidence=0.8)
        print(f"   Result: {'✅ Success' if success else '❌ Failed'}")
        time.sleep(0.5)  # Small delay between tests
    
    # Show statistics
    stats = manager.get_statistics()
    print(f"\n📊 Statistics:")
    print(f"   Total actions: {stats['total_actions']}")
    print(f"   Success rate: {stats['success_rate']:.2%}")
    
    print("\n🎉 Computer control test completed!")

if __name__ == "__main__":
    main()
