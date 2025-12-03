import math
from typing import Optional

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.messages.flat.QuickChatSelection import QuickChatSelection
from rlbot.utils.structures.ball_prediction_struct import BallPrediction, Slice
from rlbot.utils.structures.game_data_struct import GameTickPacket

from util.boost_pad_tracker import BoostPadTracker, BoostPad
from util.drive import steer_toward_target
from util.orientation import Orientation, relative_location
from util.sequence import Sequence, ControlStep
from util.vec import Vec3


class MyBot(BaseAgent):

    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.active_sequence: Sequence = None
        self.boost_pad_tracker = BoostPadTracker()

    def initialize_agent(self):
        # Set up information about the boost pads now that the game is active and the info is available
        self.boost_pad_tracker.initialize_boosts(self.get_field_info())

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        """
        This function will be called by the framework many times per second. This is where you can
        see the motion of the ball, etc. and return controls to drive your car.
        """

        # Keep our boost pad info updated with which pads are currently active
        self.boost_pad_tracker.update_boost_status(packet)

        # This is good to keep at the beginning of get_output. It will allow you to continue
        # any sequences that you may have started during a previous call to get_output.
        if self.active_sequence is not None and not self.active_sequence.done:
            controls = self.active_sequence.tick(packet)
            if controls is not None:
                return controls

        # Gather some information about our car and the ball
        my_car = packet.game_cars[self.index]
        car_location = Vec3(my_car.physics.location)
        car_velocity = Vec3(my_car.physics.velocity)
        ball_location = Vec3(packet.game_ball.physics.location)
        ball_velocity = Vec3(packet.game_ball.physics.velocity)
        car_orientation = Orientation(my_car.physics.rotation)

        controls = SimpleControllerState()

        if not my_car.has_wheel_contact and car_location.z < 200:
            controls.roll = -car_orientation.right.z
            controls.pitch = -car_orientation.forward.z
            controls.throttle = 0.5
            return controls

        # Handle kickoffs with a strong dash toward the ball.
        if self.is_kickoff(packet):
            controls.steer = steer_toward_target(my_car, ball_location)
            controls.throttle = 1.0
            controls.boost = abs(controls.steer) < 0.2
            if car_location.flat().dist(ball_location.flat()) < 1600 and self.active_sequence is None:
                return self.begin_front_flip(packet)
            return controls

        ball_prediction = self.get_ball_prediction_struct()
        intercept = self.select_intercept_slice(ball_prediction, my_car, packet.game_info.seconds_elapsed)
        target_slice = intercept if intercept is not None else None
        target_position = Vec3(target_slice.physics.location) if target_slice else ball_location

        # Approach from behind the ball relative to the opponent's goal like Nexto.
        opponent_goal_y = 5120 if self.team == 0 else -5120
        opponent_goal = Vec3(0, opponent_goal_y, 0)
        own_goal = Vec3(0, -opponent_goal_y, 0)
        shadow_target = self.choose_shadow_position(ball_location, ball_velocity, own_goal)
        if shadow_target is not None:
            target_position = shadow_target

        approach_goal = opponent_goal if shadow_target is None else own_goal
        approach_direction = (approach_goal - target_position).flat()
        if approach_direction.length() < 1:
            approach_direction = Vec3(0, 1 if opponent_goal_y > 0 else -1, 0)
        offset_distance = 320 if target_slice and target_slice.physics.location.z < 150 else 240
        target_location = target_position - approach_direction.rescale(offset_distance)
        target_location.z = max(0, target_location.z)

        chosen_boost = None
        if my_car.boost < 25 and not self.is_kickoff(packet):
            chosen_boost = self.choose_best_boost(car_location, target_location, ball_location, own_goal)
            if chosen_boost is not None:
                target_location = chosen_boost.location
                target_slice = None
                time_remaining = None

        # Visualize our intent.
        self.renderer.draw_line_3d(car_location, target_location, self.renderer.white())
        self.renderer.draw_rect_3d(target_location, 10, 10, True, self.renderer.cyan(), centered=True)
        self.renderer.draw_line_3d(target_position, target_position + Vec3(0, 0, 200), self.renderer.red())

        relative_target = relative_location(car_location, car_orientation, target_location)
        angle_to_target = math.atan2(relative_target.y, relative_target.x)
        distance_to_target = car_location.flat().dist(target_location.flat())

        controls.steer = steer_toward_target(my_car, target_location)

        time_remaining = None
        if target_slice is not None:
            time_remaining = target_slice.game_seconds - packet.game_info.seconds_elapsed

        desired_speed = self.choose_desired_speed(distance_to_target, angle_to_target, time_remaining)
        forward_speed = car_velocity.dot(car_orientation.forward)
        speed_error = desired_speed - forward_speed
        controls.throttle = max(min(speed_error / 500, 1.0), -1.0)
        controls.boost = speed_error > 400 and abs(angle_to_target) < 0.35 and desired_speed > 1200

        # Apply gentle steering help to stay aligned when making tight turns.
        controls.handbrake = abs(angle_to_target) > 1.9 and car_velocity.length() < 1000

        # Nudge the ball up for a dribble when we're under it and it's low.
        relative_ball = relative_location(car_location, car_orientation, ball_location)
        low_and_centered = relative_ball.x > -30 and abs(relative_ball.y) < 150 and ball_location.z < 200
        if my_car.has_wheel_contact and distance_to_target < 160 and ball_location.z < 120:
            controls.jump = True
        elif low_and_centered and abs(angle_to_target) < 0.35:
            controls.pitch = -0.15

        return controls

    def choose_shadow_position(self, ball_location: Vec3, ball_velocity: Vec3, own_goal: Vec3) -> Optional[Vec3]:
        ball_moving_toward_goal = (own_goal.y - ball_location.y) * ball_velocity.y > 0 and abs(ball_velocity.y) > 150
        if not ball_moving_toward_goal:
            return None

        ball_distance_from_goal = ball_location.flat().dist(own_goal.flat())
        if ball_distance_from_goal > 3800:
            return None

        goal_to_ball = (ball_location - own_goal).flat()
        if goal_to_ball.length() < 1:
            return None

        shadow_distance = max(650, min(1200, ball_distance_from_goal * 0.45))
        shadow_direction = goal_to_ball.rescale(1)
        target = ball_location - shadow_direction.rescale(shadow_distance)
        target.z = 0
        return target

    def choose_best_boost(self, car_location: Vec3, current_target: Vec3, ball_location: Vec3, own_goal: Vec3) -> Optional[BoostPad]:
        critical_defense = ball_location.flat().dist(own_goal.flat()) < 1800
        best_pad: Optional[BoostPad] = None
        best_score = None
        for pad in self.boost_pad_tracker.get_full_boosts():
            if not pad.is_active:
                continue

            pad_distance = car_location.flat().dist(pad.location.flat())
            if pad_distance > 5000:
                continue

            if critical_defense and pad_distance > 1200:
                continue

            detour_score = pad_distance + pad.location.flat().dist(current_target.flat()) * 0.35
            if best_score is None or detour_score < best_score:
                best_score = detour_score
                best_pad = pad

        return best_pad

    def begin_front_flip(self, packet):
        # Send some quickchat just for fun
        self.send_quick_chat(team_only=False, quick_chat=QuickChatSelection.Information_IGotIt)

        # Do a front flip. We will be committed to this for a few seconds and the bot will ignore other
        # logic during that time because we are setting the active_sequence.
        self.active_sequence = Sequence([
            ControlStep(duration=0.05, controls=SimpleControllerState(jump=True)),
            ControlStep(duration=0.05, controls=SimpleControllerState(jump=False)),
            ControlStep(duration=0.2, controls=SimpleControllerState(jump=True, pitch=-1)),
            ControlStep(duration=0.8, controls=SimpleControllerState()),
        ])

        # Return the controls associated with the beginning of the sequence so we can start right away.
        return self.active_sequence.tick(packet)

    def is_kickoff(self, packet: GameTickPacket) -> bool:
        ball = packet.game_ball
        return packet.game_info.is_round_active and ball.physics.location.x == 0 and ball.physics.location.y == 0

    def select_intercept_slice(self, ball_prediction: BallPrediction, car, current_time: float):
        if ball_prediction is None:
            return None

        car_location = Vec3(car.physics.location)
        car_speed = Vec3(car.physics.velocity).length()

        # Look a few seconds into the future, skipping slices for efficiency.
        for i in range(0, ball_prediction.num_slices, 8):
            ball_slice: Slice = ball_prediction.slices[i]
            target = Vec3(ball_slice.physics.location)
            if target.z > 250:
                continue

            travel_time = self.estimate_travel_time(car_location, car_speed, target)
            time_available = ball_slice.game_seconds - current_time
            if travel_time < time_available + 0.1:
                return ball_slice

        return None

    @staticmethod
    def estimate_travel_time(car_location: Vec3, car_speed: float, target: Vec3) -> float:
        distance = car_location.flat().dist(target.flat())
        max_speed = 2300
        effective_speed = max(700, min(max_speed, car_speed + 900))
        return distance / effective_speed

    @staticmethod
    def choose_desired_speed(distance: float, angle_to_target: float, time_remaining: Optional[float]) -> float:
        if time_remaining is not None and time_remaining > 0:
            return max(700, min(distance / max(time_remaining, 0.1), 2200))

        base_speed = 1900 if abs(angle_to_target) < 0.25 else 1500
        if distance < 800:
            return max(700, distance * 1.2)
        return base_speed
