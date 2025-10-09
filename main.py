import cv2
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

def read_omr_advanced(image_path):
    """
    Đọc phiếu trắc nghiệm phức tạp (50 câu hoặc nhiều hơn)
    Hỗ trợ: phiếu 2 cột, có số báo danh, mã đề thi
    """

    print(f"Đọc ảnh: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Không thể đọc file: {image_path}")
        return None

    print(f"Kích thước ảnh: {image.shape[1]}x{image.shape[0]} pixels")

    # Tiền xử lý ảnh
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Cân bằng histogram để cải thiện độ tương phản
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Làm mờ nhẹ
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Threshold adaptive
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Morphological operations để làm sạch
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Hiển thị ảnh xử lý với kích thước gốc
    display = image.copy()
    # cv2.imshow("Original", image)
    # cv2.imshow("Processed", thresh)

    # Tìm contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Phát hiện vòng tròn
    circles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # Lọc theo diện tích - linh hoạt cho nhiều kích thước
        if 80 < area < 2000:
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)

                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h) if h > 0 else 0

                # Điều kiện: tròn và tỷ lệ gần vuông
                if circularity > 0.6 and 0.75 <= aspect_ratio <= 1.25:
                    if 8 <= w <= 60 and 8 <= h <= 60:
                        circles.append({
                            'contour': contour,
                            'x': x,
                            'y': y,
                            'w': w,
                            'h': h,
                            'area': area,
                            'circularity': circularity,
                            'center': (x + w//2, y + h//2)
                        })

    if len(circles) < 4:
        print("❌ Không đủ vòng tròn để phân tích!")
        return None

    # PHÂN TÍCH CẤU TRÚC LƯỚI
    # Nhóm theo cột X
    all_x = sorted([c['center'][0] for c in circles])
    x_groups = []
    x_tolerance = 12

    for x in all_x:
        added = False
        for group in x_groups:
            if abs(x - np.mean(group)) <= x_tolerance:
                group.append(x)
                added = True
                break
        if not added:
            x_groups.append([x])

    # Lấy các cột có >= 5 vòng tròn
    column_positions = sorted([np.mean(g) for g in x_groups if len(g) >= 5])

    # PHÂN LOẠI CÁC VÙNG
    # Tìm nhóm 4-5 cột liên tiếp (đáp án A,B,C,D hoặc A,B,C,D,E)
    answer_column_groups = []
    i = 0
    while i < len(column_positions):
        group = [column_positions[i]]
        j = i + 1
        while j < len(column_positions):
            spacing = column_positions[j] - group[-1]
            if spacing < 60:  # Khoảng cách giữa các cột đáp án
                group.append(column_positions[j])
                j += 1
            else:
                break

        if 4 <= len(group) <= 5:  # Nhóm đáp án hợp lệ
            answer_column_groups.append(group)
            i = j
        else:
            i += 1

    if len(answer_column_groups) == 0:
        print("❌ Không tìm thấy vùng đáp án!")
        return None

    # Lọc circles chỉ giữ các vòng tròn trong vùng đáp án
    valid_circles = []
    for circle in circles:
        cx = circle['center'][0]
        for group in answer_column_groups:
            # Kiểm tra vòng tròn có nằm trong nhóm cột này không
            if group[0] - 20 <= cx <= group[-1] + 20:
                # Xác định cột trong nhóm (tìm cột gần nhất)
                distances = [abs(cx - col_x) for col_x in group]
                min_distance = min(distances)

                # CHỈ CHẤP NHẬN nếu GẦN cột (< 12px) để loại bỏ các vòng tròn lạ
                if min_distance <= 12:
                    circle['column_group'] = answer_column_groups.index(group)
                    circle['column_in_group'] = distances.index(min_distance)
                    valid_circles.append(circle)
                break

    if len(valid_circles) < 4:
        print("❌ Không đủ vòng tròn trong vùng đáp án!")
        return None

    # TÍNH TOÁN ĐỘ ĐẬM CỦA TỪNG VÒNG TRÒN
    for circle in valid_circles:
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [circle['contour']], -1, 255, -1)

        pixels = gray[mask == 255]
        if len(pixels) == 0:
            circle['intensity'] = 255
            circle['dark_pixels'] = 0
            continue

        # Tính độ sáng trung bình (càng thấp = càng đậm)
        circle['intensity'] = np.mean(pixels)
        circle['min_intensity'] = np.min(pixels)
        circle['std_intensity'] = np.std(pixels)

        # Đếm số pixel RẤT TỐI (< 100) - đặc trưng của vòng tròn đã tô thật sự
        very_dark_pixels = np.sum(pixels < 100)
        circle['very_dark_ratio'] = very_dark_pixels / len(pixels)

        # Đếm pixel tối (< 150)
        dark_pixels = np.sum(pixels < 150)
        circle['dark_ratio'] = dark_pixels / len(pixels)

    # PHÂN BIỆT PHIẾU 1 CỘT vs NHIỀU CỘT
    # Nếu có nhiều nhóm cột đáp án -> phiếu nhiều cột (đọc theo cột)
    # Nếu chỉ có 1 nhóm -> phiếu đơn giản (đọc theo hàng)

    if len(answer_column_groups) > 1:
        # PHIẾU NHIỀU CỘT - ĐỌC THEO CỘT (top-to-bottom, left-to-right)
        answers = []

        for group_idx, group_cols in enumerate(answer_column_groups):
            # Lấy tất cả circles trong nhóm cột này
            group_circles = [c for c in valid_circles if c.get('column_group') == group_idx]

            # Sắp xếp theo Y (từ trên xuống)
            group_circles.sort(key=lambda c: c['y'])

            # Nhóm theo hàng
            rows = []
            current_row = []
            y_tolerance = 20

            for circle in group_circles:
                if len(current_row) == 0:
                    current_row.append(circle)
                else:
                    if abs(circle['y'] - current_row[0]['y']) <= y_tolerance:
                        current_row.append(circle)
                    else:
                        if len(current_row) >= 3:
                            rows.append(current_row)
                        current_row = [circle]

            if len(current_row) >= 3:
                rows.append(current_row)

            # Xử lý từng hàng trong cột này
            for row in rows:
                row.sort(key=lambda c: c['column_in_group'])

                # Đảm bảo chỉ có 1 vòng tròn mỗi cột
                unique_circles = {}
                for circle in row:
                    col_idx = circle['column_in_group']
                    if col_idx not in unique_circles:
                        unique_circles[col_idx] = circle

                row_circles = sorted(unique_circles.values(), key=lambda c: c['column_in_group'])

                if len(row_circles) < 3:
                    continue

                # Tìm ô đậm nhất
                darkest = min(row_circles, key=lambda c: c['intensity'])
                lightest = max(row_circles, key=lambda c: c['intensity'])

                intensities = [c['intensity'] for c in row_circles]
                avg_intensity = np.mean(intensities)
                std_intensity = np.std(intensities)
                intensity_range = lightest['intensity'] - darkest['intensity']

                # Kiểm tra có đáp án rõ ràng không
                # Giảm threshold cho phù hợp với nhiều loại phiếu
                has_clear_difference = intensity_range > 6  # Giảm từ 25 -> 6
                is_much_darker = (avg_intensity - darkest['intensity']) > 3  # Giảm từ 18 -> 3
                has_variation = std_intensity > 2  # Giảm từ 12 -> 2

                if (has_clear_difference and is_much_darker) or (has_variation and is_much_darker):
                    answer_index = darkest['column_in_group']

                    # Vẽ lên ảnh
                    x, y, w, h = darkest['x'], darkest['y'], darkest['w'], darkest['h']
                    cv2.drawContours(display, [darkest['contour']], -1, (0, 0, 255), 2)
                    cv2.circle(display, darkest['center'], 3, (0, 255, 0), -1)

                    letter_map = ['A', 'B', 'C', 'D', 'E']
                    letter = letter_map[answer_index] if answer_index < len(letter_map) else '?'

                    question_num = len(answers) + 1
                    cv2.putText(display, f"{question_num}:{letter}", (x-30, y+h//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

                    answers.append(answer_index)

    else:
        # PHIẾU ĐƠN GIẢN 1 CỘT - ĐỌC THEO HÀNG
        valid_circles.sort(key=lambda c: c['y'])

        rows = []
        current_row = []
        y_tolerance = 20

        for circle in valid_circles:
            if len(current_row) == 0:
                current_row.append(circle)
            else:
                if abs(circle['y'] - current_row[0]['y']) <= y_tolerance:
                    current_row.append(circle)
                else:
                    if len(current_row) >= 3:
                        rows.append(current_row)
                    current_row = [circle]

        if len(current_row) >= 3:
            rows.append(current_row)

        answers = []

        for row_idx, row in enumerate(rows):
            # Nhóm theo column_group
            groups = {}
            for circle in row:
                group_id = circle.get('column_group', 0)
                if group_id not in groups:
                    groups[group_id] = []
                groups[group_id].append(circle)

            # Xử lý từng nhóm
            for group_id, group_circles in sorted(groups.items()):
                if len(group_circles) < 3:
                    continue

                # Sắp xếp theo cột trong nhóm
                group_circles.sort(key=lambda c: c['column_in_group'])

                # ĐẢM BẢO CHỈ CÓ 1 VÒNG TRÒN MỖI CỘT
                unique_circles = {}
                for circle in group_circles:
                    col_idx = circle['column_in_group']
                    if col_idx not in unique_circles:
                        unique_circles[col_idx] = circle

                # Chuyển về danh sách và sắp xếp lại
                group_circles = sorted(unique_circles.values(), key=lambda c: c['column_in_group'])

                # Bỏ qua nếu không đủ 3-5 cột
                if len(group_circles) < 3:
                    continue

                # Tìm ô đậm nhất VÀ ô sáng nhất
                darkest = min(group_circles, key=lambda c: c['intensity'])
                lightest = max(group_circles, key=lambda c: c['intensity'])

                # Kiểm tra xem có đậm hơn đáng kể so với các ô khác không
                intensities = [c['intensity'] for c in group_circles]
                avg_intensity = np.mean(intensities)
                std_intensity = np.std(intensities)
                intensity_range = lightest['intensity'] - darkest['intensity']

                # TIÊU CHÍ MỚI - SO SÁNH TƯƠNG ĐỐI
                # Giảm threshold cho phù hợp với nhiều loại phiếu
                has_clear_difference = intensity_range > 6  # Giảm từ 25 -> 6
                is_much_darker = (avg_intensity - darkest['intensity']) > 3  # Giảm từ 18 -> 3
                has_variation = std_intensity > 2  # Giảm từ 12 -> 2

                # Chỉ chấp nhận nếu có SỰ KHÁC BIỆT RÕ RÀNG
                if (has_clear_difference and is_much_darker) or (has_variation and is_much_darker):
                    answer_index = darkest['column_in_group']

                    # Vẽ lên ảnh
                    x, y, w, h = darkest['x'], darkest['y'], darkest['w'], darkest['h']
                    cv2.drawContours(display, [darkest['contour']], -1, (0, 0, 255), 2)
                    cv2.circle(display, darkest['center'], 3, (0, 255, 0), -1)

                    letter_map = ['A', 'B', 'C', 'D', 'E']
                    letter = letter_map[answer_index] if answer_index < len(letter_map) else '?'

                    question_num = len(answers) + 1
                    cv2.putText(display, f"{question_num}:{letter}", (x-30, y+h//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

                    answers.append(answer_index)

    # Trả về cả answers và display image
    return answers, image, display


def main():
    # Nhập file
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Nhập tên file ảnh: ").strip()

    if not os.path.exists(image_path):
        print(f"❌ File không tồn tại: {image_path}")
        return

    # Đọc phiếu
    result = read_omr_advanced(image_path)

    if result is None:
        print("\n❌ Không đọc được đáp án")
        return

    answers, original_img, marked_img = result

    if answers and len(answers) > 0:
        # Chuyển đổi thành chữ cái
        letter_map = ['A', 'B', 'C', 'D', 'E']
        result_letters = [letter_map[ans] if 0 <= ans < len(letter_map) else '?' for ans in answers]

        # In kết quả đơn giản
        print("\n✅ ĐÁP ÁN:")
        print(result_letters)

        # Hiển thị ảnh gốc và ảnh đã khoanh bằng matplotlib
        fig, axes = plt.subplots(1, 2, figsize=(16, 10))

        # Chuyển từ BGR sang RGB cho matplotlib
        original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        marked_rgb = cv2.cvtColor(marked_img, cv2.COLOR_BGR2RGB)

        axes[0].imshow(original_rgb)
        axes[0].set_title('Ảnh gốc', fontsize=14)
        axes[0].axis('off')

        axes[1].imshow(marked_rgb)
        axes[1].set_title('Ảnh đã khoanh đáp án', fontsize=14)
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()
    else:
        print("\n❌ Không đọc được đáp án")


if __name__ == "__main__":
    main()
