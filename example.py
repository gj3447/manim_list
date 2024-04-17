from manim import *
import numpy as np
from manim.utils.color import *

class CreateCircle(Scene):
    def construct(self):
        circle = Circle()  # 원을 생성
        self.play(Create(circle))  # 원을 화면에 그리기
        self.wait(1)  # 1초 동안 대기

class FFTVisualization(Scene):
    def construct(self):
        # Sample data
        t = np.linspace(0, 2*np.pi, 200)
        y = np.sin(2 * np.pi * t) + 0.5 * np.sin(8 * np.pi * t)

        # Create a line graph for the signal
        signal_graph = self.create_line_graph(t, y, color=BLUE)
        signal_graph.scale(0.1)  # Scale down
        signal_graph.move_to(ORIGIN)  # Move to screen center after scaling
        self.play(Create(signal_graph))
        self.wait(1)

        # Perform FFT
        freqs = np.fft.fftfreq(len(t), d=(t[1] - t[0]))
        magnitudes = np.abs(np.fft.fft(y))

        # Graph of the frequency domain
        freq_graph = self.create_line_graph(freqs, magnitudes, color=RED)
        freq_graph.scale(0.01)  # Scale down     
        freq_graph.move_to(ORIGIN)  
          # Ensure it's centered
        self.play(Transform(signal_graph, freq_graph))
        self.wait(1)

    def create_line_graph(self, x, y, color):
        graph = VMobject()
        graph.set_points_smoothly([*[np.array([xi, yi, 0]) for xi, yi in zip(x, y)]])
        graph.set_color(color)
        return graph
    
class LogisticMapScene(Scene):
    def construct(self):
        axes = Axes(
            x_range=[2, 4, 0.5],
            y_range=[0, 1, 0.2],
            x_length=7,
            y_length=5,
            axis_config={"color": BLUE},
        )

        # Axes labels
        labels = axes.get_axis_labels(x_label="r", y_label="x")
        self.play(Create(axes), Write(labels))

        # Bifurcation diagram
        r_values = np.linspace(2.5, 4, 150)
        iterations = 1000
        last = 100

        for r in r_values:
            x = 0.5
            for _ in range(iterations):
                x = r * x * (1 - x)
            self.add(Dot(point=axes.coords_to_point(r, x), color=YELLOW, radius=0.02))

        self.wait(1)
class MatrixTransformation(Scene):
    def construct(self):
        # 초기 벡터를 설정합니다.
        grid = NumberPlane()
        vector = Arrow(start=ORIGIN, end=[2, 2, 0], buff=0, color=BLUE)
        matrix = [[1, 1], [0, 1]]  # 선형 변환을 위한 매트릭스

        self.play(Create(grid))  # 그리드 생성
        self.play(Create(vector))  # 초기 벡터 그리기
        self.wait(1)

        # 벡터에 매트릭스를 적용합니다.
        transformed_vector = vector.copy()
        transformed_vector.apply_matrix(matrix)
        transformed_vector.set_color(RED)

        # 변환된 벡터를 화면에 표시합니다.
        self.play(Transform(vector, transformed_vector))
        self.wait(1)

        # 벡터와 그리드를 제거합니다.
        self.play(FadeOut(vector), FadeOut(grid))
        self.wait(1)
class CoordinateTransformation(Scene):
    def construct(self):
        # 좌표계와 원래 벡터 생성
        plane = NumberPlane()
        vector = Arrow(start=ORIGIN, end=[2, 2, 0], buff=0, color=BLUE, tip_length=0.2)
        label = MathTex("v").next_to(vector.get_end(), RIGHT+UP)

        # 그리드와 벡터를 화면에 추가
        self.play(Create(plane), Create(vector), Write(label))
        self.wait(1)

        # 선형 변환 매트릭스 정의
        matrix = [[1, 1], [0, 1]]  # Shear 변환

        # 좌표계에 변환 적용
        transformed_plane = plane.copy()
        transformed_plane.apply_matrix(matrix)
        transformed_plane.set_color(RED)

        # 벡터에도 같은 변환 적용
        transformed_vector = vector.copy()
        transformed_vector.apply_matrix(matrix)
        transformed_vector.set_color(RED)
        transformed_label = MathTex("Av").next_to(transformed_vector.get_end(), RIGHT+UP)

        # 변환된 그리드와 벡터를 화면에 표시
        self.play(Transform(plane, transformed_plane),
                  Transform(vector, transformed_vector),
                  Transform(label, transformed_label))
        self.wait(1)

        # 최종 상태에서 모든 요소 제거
        self.play(FadeOut(plane), FadeOut(vector), FadeOut(label))
        self.wait(1)

class RiemannZetaFunction(Scene):
    def construct(self):
        plane = ComplexPlane(
            x_range=(-2, 2, 1),
            y_range=(-2, 2, 1),
            background_line_style={"stroke_width": 1, "stroke_color": BLUE}
        ).add_coordinates()

        self.add(plane)

        step = 0.1
        for real in np.arange(-2, 2, step):
            for imag in np.arange(-2, 2, step):
                s = complex(real, imag)
                zeta_value = self.zeta(s)
                color = self.color_map(zeta_value)
                dot = Dot(plane.n2p(s), color=color, radius=0.05)
                self.add(dot)

        self.wait()
        
    def zeta(self, s, terms=100):
        return sum([1/n**s for n in range(1, terms)])

    def color_map(self, value):
        hue = (np.angle(value) / (2 * np.pi)) % 1
        saturation = 1
        lightness = min(np.abs(value), 1)
        return interpolate_color(WHITE, BLUE, lightness)  # 수정된 부분
    
class RiemannZetaTransformation(Scene):
    def zeta(self, s, terms=50):
        # Compute the Riemann Zeta function for a given s using 50 terms
        return sum([1/np.power(n, s) for n in range(1, terms)])
    
    def construct(self):
        plane = ComplexPlane(x_range=[-2, 2], y_range=[-2, 2])
        self.add(plane)

        # Grid of points
        points = [x + y*1j for x in np.linspace(-2, 2, 10) for y in np.linspace(-2, 2, 10)]

        # Original points
        dots = VGroup(*[Dot(plane.coords_to_point(p.real, p.imag), color=BLUE) for p in points])
        self.add(dots)

        # Transformed points using the Riemann Zeta function
        transformed_dots = VGroup()
        for p in points:
            zeta_val = self.zeta(p)
            if np.isfinite(zeta_val.real) and np.isfinite(zeta_val.imag):
                dot = Dot(plane.coords_to_point(zeta_val.real, zeta_val.imag), color=RED)
                transformed_dots.add(dot)

        # Add transformed points to the scene
        self.play(Transform(dots, transformed_dots))
        self.wait(1)

        # Optionally, connect original and transformed points with arrows
        for original, transformed in zip(dots, transformed_dots):
            arrow = Arrow(original.get_center(), transformed.get_center(), buff=0.1, color=YELLOW)
            self.add(arrow)

        self.wait(2)


class RiemannZetaTransformation2(Scene):
    def zeta(self, s, terms=50):
        # 리만 제타 함수 계산
        return sum([1/np.power(n, s) for n in range(1, terms)])
    
    def construct(self):
        # 초기 좌표 평면 생성
        plane = ComplexPlane(
            x_range=[-100, 100, 0.25],  # x_range에서 0.25는 x축 간격을 의미
            y_range=[-100, 100, 0.25],  # y_range에서 0.25는 y축 간격을 의미
            axis_config={"stroke_color": BLUE},
            background_line_style={
                "stroke_color": BLUE_A,
                "stroke_width": 1,
                "stroke_opacity": 0.75  # 배경선의 투명도 조절
            }
        )
       # plane.add_coordinates(font_size =0)  # 좌표 추가
        self.add(plane)

        # 좌표 평면 복제 및 적용할 변형 생성
        transformed_plane = plane.copy()

        # 평면 위 각 점에 리만 제타 함수 적용
        def zeta_transform(point):
            x, y = point[:2]
            z = complex(x, y)
            zeta_val = self.zeta(z)
            if np.isfinite(zeta_val.real) and np.isfinite(zeta_val.imag):
                return np.array([zeta_val.real, zeta_val.imag, 0])
            else:
                return point

        # 함수 적용을 위해 평면의 모든 점에 변형 적용
        transformed_plane.apply_function(zeta_transform)
        transformed_plane.set_color(RED)

        # 평면 변형 애니메이션 실행
        self.play(Transform(plane, transformed_plane), run_time=5)  # 애니메이션 시간을 5초로 설정
        self.wait(1)

        # 추가: 원래 평면과 변형된 평면을 함께 표시
        self.add(plane.copy().set_color(BLUE), transformed_plane)
        self.wait(2)

# 저장 및 실행
class RiemannZetaTransformation3(Scene):
    def zeta(self, s, terms=50):
        # 리만 제타 함수 계산
        return sum([1/np.power(n, s) for n in range(1, terms)])

    def construct(self):
        # 눈금 레이블 없이 촘촘한 격자와 함께 초기 좌표 평면 생성
        plane = ComplexPlane(
            x_range=[-2, 2, 0.01],  # x_range에서 0.25는 x축 간격을 의미
            y_range=[-2, 2, 0.01],  # y_range에서 0.25는 y축 간격을 의미
            axis_config={
                "stroke_color": BLUE,# 눈금 제거
            },
            background_line_style={
                "stroke_color": BLUE_A,
                "stroke_width": 1,
                "stroke_opacity": 1  # 배경선의 투명도 조절
            },
            faded_line_ratio=0  # 페이드된 선 없음
        )
        # 눈금과 숫자 레이블을 추가하지 않음
        self.add(plane)

        # 좌표 평면 복제 및 적용할 변형 생성
        transformed_plane = plane.copy()

        # 평면 위 각 점에 리만 제타 함수 적용
        def zeta_transform(point):
            x, y = point[:2]
            z = complex(x, y)
            zeta_val = self.zeta(z)
            if np.isfinite(zeta_val.real) and np.isfinite(zeta_val.imag):
                return np.array([zeta_val.real, zeta_val.imag, 0])
            else:
                return point

        # 함수 적용을 위해 평면의 모든 점에 변형 적용
        transformed_plane.apply_function(zeta_transform)
        transformed_plane.set_color(RED)

        # 평면 변형 애니메이션 실행
        self.play(Transform(plane, transformed_plane), run_time=100)  # 애니메이션 시간을 5초로 설정
        self.wait(1)

        # 추가: 원래 평면과 변형된 평면을 함께 표시
        self.add(plane.copy().set_color(BLUE), transformed_plane)
        self.wait(2)
