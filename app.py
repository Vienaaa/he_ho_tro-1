# Import các thư viện cần thiết
from flask import Flask, render_template, request, redirect, url_for, send_file, session, Response, make_response
import numpy as np
import pandas as pd
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Đặt backend Agg
import matplotlib.pyplot as plt
import io
import base64


# Khởi tạo ứng dụng Flask
app = Flask(__name__)
# KHÓA BÍ MẬT QUAN TRỌNG: Thay thế bằng một chuỗi ngẫu nhiên, phức tạp và bí mật của riêng bạn trong môi trường sản xuất!
app.secret_key = 'your_super_secret_and_complex_key_here_please_change_me'

# Danh sách mặc định tiêu chí và phương án (dùng cho chọn ở bước 2)
AVAILABLE_CRITERIA = [
    "Chi phí & mức phí bảo hiểm",
    "Phạm vi bảo hiểm",
    "Uy tín & đánh giá",
    "Dịch vụ khách hàng",
    "Điều khoản hợp đồng",
    "Khả năng chi trả",
    "Mức độ linh hoạt",
    "Tốc độ giải quyết bồi thường",
    "Sản phẩm đa dạng",
    "Mạng lưới bệnh viện/phòng khám"
]

AVAILABLE_OPTIONS = [
    "Bảo Việt Nhân Thọ",
    "Prudential Việt Nam",
    "Dai-ichi Việt Nam",
    "Hanwha Life Việt Nam",
    "Manulife Việt Nam",
    "Generali Việt Nam",
    "Fubon Life Việt Nam",
    "Chubb Life Việt Nam",
    "Cathay Life Việt Nam",
    "Sun Life Việt Nam"
]

# Hàm tính trọng số eigenvector và Consistency Ratio (CR)
def calculate_weights(matrix):
    """
    Tính toán trọng số (eigenvector) và giá trị riêng lớn nhất (lambda_max) từ ma trận so sánh cặp.
    Parameters:
    matrix (np.array): Ma trận so sánh cặp.
    Returns:
    tuple: (weights, max_eigval) - trọng số và giá trị riêng lớn nhất.
    """
    # Tính toán giá trị riêng và vector riêng
    eigvals, eigvecs = np.linalg.eig(matrix)
    
    # Tìm chỉ số của giá trị riêng lớn nhất (phần thực)
    max_index = np.argmax(eigvals.real)
    max_eigval = eigvals.real[max_index]
    
    # Lấy vector riêng tương ứng và chuẩn hóa
    weights = eigvecs[:, max_index].real
    weights = weights / np.sum(weights)
    
    return weights, max_eigval

def consistency_ratio(matrix):
    """
    Tính toán chỉ số nhất quán (Consistency Ratio - CR) của ma trận.
    Parameters:
    matrix (np.array): Ma trận so sánh cặp.
    Returns:
    float: Giá trị CR.
    """
    n = matrix.shape[0]
    
    # Nếu n=1 hoặc 2, CR luôn bằng 0 (không cần tính toán)
    if n <= 2:
        return 0.0

    weights, max_eigval = calculate_weights(matrix)
    
    # Tính Consistency Index (CI)
    ci = (max_eigval - n) / (n - 1)
    
    # Bảng RI (Random Index) cho n từ 1 đến 10 (theo chuẩn AHP)
    ri_dict = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    ri = ri_dict.get(n, 1.49) # Nếu n > 10, lấy RI = 1.49

    # Tránh chia cho 0 nếu RI bằng 0 (chỉ xảy ra khi n <= 2, đã xử lý ở trên)
    if ri == 0:
        return 0.0
        
    cr = ci / ri
    return cr

# Route trang chủ
@app.route('/')
def home():
    return render_template('home.html')

# Bước 1: Nhập số lượng tiêu chí và phương án
@app.route('/step1', methods=['GET', 'POST'])
def step1():
    error = None
    if request.method == 'POST':
        try:
            n_criteria = int(request.form['n_criteria'])
            n_options = int(request.form['n_options'])
            if n_criteria < 3 or n_options < 3: # AHP thường yêu cầu n >= 3 để tính CR
                error = "Phải chọn số lượng tiêu chí và phương án từ 3 trở lên."
            else:
                session['n_criteria'] = n_criteria
                session['n_options'] = n_options
                # Đặt lại các lựa chọn và ma trận nếu người dùng quay lại bước 1
                session.pop('criteria', None)
                session.pop('options', None)
                session.pop('criteria_matrix', None)
                session.pop('option_matrices', None)
                session.pop('criteria_weights', None)
                session.pop('criteria_cr', None)
                session.pop('df_results', None)
                return redirect(url_for('step2'))
        except ValueError:
            error = "Vui lòng nhập số nguyên hợp lệ."
    return render_template('step1.html', error=error)

# Bước 2: Chọn tiêu chí và phương án cụ thể
@app.route('/step2', methods=['GET', 'POST'])
def step2():
    error = None
    n_criteria = session.get('n_criteria')
    n_options = session.get('n_options')

    # Nếu chưa có dữ liệu từ bước 1, chuyển hướng về bước 1
    if n_criteria is None or n_options is None:
        return redirect(url_for('step1'))

    if request.method == 'POST':
        try:
            selected_criteria = request.form.getlist('criteria[]')
            selected_options = request.form.getlist('options[]')

            # Kiểm tra số lượng và tính duy nhất của các lựa chọn
            if len(set(selected_criteria)) != n_criteria:
                error = f"Bạn phải chọn đủ {n_criteria} tiêu chí và không được lặp."
            elif len(set(selected_options)) != n_options:
                error = f"Bạn phải chọn đủ {n_options} phương án và không được lặp."
            else:
                session['criteria'] = selected_criteria
                session['options'] = selected_options
                return redirect(url_for('step3'))
        except Exception:
            error = "Dữ liệu không hợp lệ, vui lòng thử lại."

    return render_template(
        'step2.html',
        error=error,
        n_criteria=n_criteria,
        n_options=n_options,
        available_criteria=AVAILABLE_CRITERIA,
        available_options=AVAILABLE_OPTIONS,
        # Giữ lại các lựa chọn đã có nếu người dùng quay lại bước này
        selected_criteria=session.get('criteria', []),
        selected_options=session.get('options', [])
    )

# Bước 3: Nhập ma trận so sánh cặp cho tiêu chí
@app.route('/step3', methods=['GET', 'POST'])
def step3():
    n_criteria = session.get('n_criteria')
    criteria = session.get('criteria')

    if not n_criteria or not criteria:
        return redirect(url_for('step1'))

    error = None
    matrix_display = np.eye(n_criteria).tolist() # Khởi tạo ma trận đơn vị để hiển thị ban đầu

    if request.method == 'POST':
        try:
            matrix = np.eye(n_criteria) # Khởi tạo ma trận đơn vị
            for i in range(n_criteria):
                for j in range(i + 1, n_criteria): # Chỉ lặp qua nửa trên đường chéo chính
                    # Lấy giá trị từ form
                    val_str = request.form.get(f'criteria_{i}_{j}', '1')
                    if not val_str: # Xử lý trường hợp người dùng để trống
                        raise ValueError(f"Vui lòng nhập giá trị cho ô ({criteria[i]}, {criteria[j]})")
                    
                    val = float(val_str)
                    if val == 0: # Tránh lỗi chia cho 0
                        raise ValueError("Giá trị không được bằng 0.")
                    
                    matrix[i, j] = val
                    matrix[j, i] = 1 / val # Tự động tính toán giá trị đối xứng
            
            # Cập nhật matrix_display để hiển thị lại các giá trị đã nhập
            matrix_display = matrix.tolist()

            cr = consistency_ratio(matrix)
            if cr > 0.1:
                error = f"Chỉ số nhất quán CR={cr:.3f} > 0.1. Vui lòng nhập lại ma trận."
            else:
                weights, _ = calculate_weights(matrix)
                session['criteria_matrix'] = matrix.tolist()
                session['criteria_weights'] = weights.tolist()
                session['criteria_cr'] = cr
                return redirect(url_for('step4'))
        except ValueError as e:
            error = f"Lỗi nhập liệu: {e}. Vui lòng nhập số hợp lệ."
        except Exception as e:
            error = f"Đã xảy ra lỗi không mong muốn: {e}. Vui lòng thử lại."

    return render_template(
        'step3.html',
        n_criteria=n_criteria,
        criteria=criteria,
        error=error,
        matrix=matrix_display # Truyền ma trận để hiển thị lại (bao gồm cả các giá trị đã nhập)
    )

# Bước 4: Nhập ma trận so sánh cặp cho các phương án theo từng tiêu chí
@app.route('/step4', methods=['GET', 'POST'])
def step4():
    n_criteria = session.get('n_criteria')
    n_options = session.get('n_options')
    criteria = session.get('criteria')
    options = session.get('options')
    
    if not (n_criteria and n_options and criteria and options):
        return redirect(url_for('step1'))
    
    error = None
    # Khởi tạo các ma trận hiển thị ban đầu hoặc từ session nếu có
    option_matrices_display = session.get('option_matrices', [[[1.0] * n_options for _ in range(n_options)] for _ in range(n_criteria)])

    if request.method == 'POST':
        try:
            option_matrices = []
            for c_index in range(n_criteria):
                matrix = np.eye(n_options) # Khởi tạo ma trận đơn vị cho từng tiêu chí
                for i in range(n_options):
                    for j in range(i + 1, n_options): # Chỉ lặp qua nửa trên đường chéo chính
                        val_str = request.form.get(f'criteria{c_index}_option_{i}_{j}', '1')
                        if not val_str:
                            raise ValueError(f"Vui lòng nhập giá trị cho ô ({options[i]}, {options[j]}) dưới tiêu chí '{criteria[c_index]}'.")
                        
                        val = float(val_str)
                        if val == 0:
                            raise ValueError(f"Giá trị không được bằng 0 cho ô ({options[i]}, {options[j]}) dưới tiêu chí '{criteria[c_index]}'.")
                        
                        matrix[i, j] = val
                        matrix[j, i] = 1 / val # Tự động tính toán giá trị đối xứng
                option_matrices.append(matrix)
            
            # Cập nhật option_matrices_display để hiển thị lại các giá trị đã nhập
            option_matrices_display = [m.tolist() for m in option_matrices]

            # Kiểm tra CR cho từng ma trận phương án ở đây
            for c_index, matrix_np in enumerate(option_matrices):
                cr = consistency_ratio(matrix_np)
                if cr > 0.1:
                    error = f"Chỉ số nhất quán CR={cr:.3f} > 0.1 cho ma trận phương án dưới tiêu chí '{criteria[c_index]}'. Vui lòng nhập lại."
                    break # Dừng lại nếu có bất kỳ ma trận nào không nhất quán
            
            if error: # Nếu có lỗi CR, hiển thị lại form
                pass # Lỗi đã được gán, hàm render_template sẽ được gọi
            else:
                session['option_matrices'] = [m.tolist() for m in option_matrices]
                return redirect(url_for('step5'))

        except ValueError as e:
            error = f"Lỗi nhập liệu: {e}. Vui lòng nhập số hợp lệ."
        except Exception as e:
            error = f"Đã xảy ra lỗi không mong muốn: {e}. Vui lòng thử lại."

    return render_template(
        'step4.html',
        n_criteria=n_criteria,
        n_options=n_options,
        criteria=criteria,
        options=options,
        error=error,
        enumerate=enumerate,
        option_matrices_display=option_matrices_display # Truyền ma trận để hiển thị lại
    )

# Bước 5: Tính toán và hiển thị kết quả cuối cùng
import base64
import io
import matplotlib.pyplot as plt

def get_plot_option_scores_pie_base64():
    option_labels = session.get('option_labels')
    option_scores = session.get('option_scores')
    if not option_labels or not option_scores:
        return None

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.Set3.colors[:len(option_scores)]
    explode = [0.1 if i == 0 else 0 for i in range(len(option_scores))]

    ax.pie(
        option_scores,
        labels=option_labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        explode=explode,
        textprops={'fontsize': 12}
    )
    ax.set_title('Phân Bố Điểm Tổng Hợp Các Phương Án', fontsize=16, pad=20)

    plt.tight_layout()
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    plt.close(fig)
    img.seek(0)

    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

# Nếu bạn có biểu đồ trọng số tiêu chí, viết tương tự hoặc bỏ nếu không cần
def get_plot_criteria_weights_pie_base64():
    criteria = session.get('criteria')
    criteria_weights = session.get('criteria_weights')
    if not criteria or not criteria_weights:
        return None

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.Pastel1.colors[:len(criteria)]
    explode = [0.1 if i == 0 else 0 for i in range(len(criteria))]

    ax.pie(
        criteria_weights,
        labels=criteria,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        explode=explode,
        textprops={'fontsize': 12}
    )
    ax.set_title('Phân Bố Trọng Số Tiêu Chí', fontsize=16, pad=20)

    plt.tight_layout()
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    plt.close(fig)
    img.seek(0)

    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

@app.route('/step5', methods=['GET', 'POST'])
def step5():
    n_criteria = session.get('n_criteria')
    n_options = session.get('n_options')
    criteria = session.get('criteria')
    options = session.get('options')
    criteria_matrix = session.get('criteria_matrix')
    option_matrices = session.get('option_matrices')

    empty_results = pd.DataFrame().to_dict(orient='records')
    empty_pairwise = {}

    if not all([n_criteria, n_options, criteria, options, criteria_matrix, option_matrices]):
        return render_template('step5.html', 
                              error="Dữ liệu không đầy đủ. Vui lòng kiểm tra lại các bước trước.", 
                              df_results=empty_results,
                              pairwise_matrices=empty_pairwise,
                              plot_option_scores_base64=None,
                              plot_criteria_weights_base64=None)

    try:
        criteria_matrix_np = np.array(criteria_matrix)
        criteria_cr = consistency_ratio(criteria_matrix_np)
        criteria_weights, lambda_max_criteria = calculate_weights(criteria_matrix_np)

        if criteria_cr > 0.1:
            return render_template('step5.html', 
                                  error=f"Chỉ số nhất quán CR của ma trận tiêu chí vượt ngưỡng: {criteria_cr:.4f}.",
                                  df_results=empty_results,
                                  pairwise_matrices=empty_pairwise,
                                  plot_option_scores_base64=None,
                                  plot_criteria_weights_base64=None)

        option_weights_per_criterion = []
        for c_index, matrix_list in enumerate(option_matrices):
            matrix_np = np.array(matrix_list)
            cr = consistency_ratio(matrix_np)
            weights, _ = calculate_weights(matrix_np)
            if cr > 0.1:
                return render_template('step5.html', 
                                      error=f"Chỉ số nhất quán CR={cr:.4f} của ma trận phương án dưới tiêu chí '{criteria[c_index]}' vượt ngưỡng.",
                                      df_results=empty_results,
                                      pairwise_matrices=empty_pairwise,
                                      plot_option_scores_base64=None,
                                      plot_criteria_weights_base64=None)
            option_weights_per_criterion.append(weights)

        option_weights_matrix = np.vstack(option_weights_per_criterion)
        overall_scores = criteria_weights @ option_weights_matrix

        df_results = pd.DataFrame({
            'Phương án': options,
            'Điểm tổng hợp': overall_scores
        }).sort_values(by='Điểm tổng hợp', ascending=False).reset_index(drop=True)
        df_results['Xếp hạng'] = df_results.index + 1
        for i, criterion in enumerate(criteria):
            df_results[criterion] = option_weights_matrix[i, :]

        # In dữ liệu để kiểm tra
        print("Criteria:", criteria)
        print("Criteria Weights:", criteria_weights.tolist())
        print("Option Labels:", df_results['Phương án'].tolist())
        print("Option Scores:", df_results['Điểm tổng hợp'].tolist())

        session['df_results'] = df_results.to_dict(orient='records')
        session['criteria'] = criteria
        session['criteria_weights'] = criteria_weights.tolist()
        session['option_labels'] = df_results['Phương án'].tolist()
        session['option_scores'] = df_results['Điểm tổng hợp'].tolist()

    except Exception as e:
        return render_template('step5.html', 
                              error=f"Lỗi khi tính toán: {e}.", 
                              df_results=empty_results,
                              pairwise_matrices=empty_pairwise,
                              plot_option_scores_base64=None,
                              plot_criteria_weights_base64=None)

    if request.method == 'POST' and 'download' in request.form:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_results.to_excel(writer, index=False, sheet_name='Kết quả AHP')
        output.seek(0)
        return send_file(output,
                        as_attachment=True,
                        download_name='ket_qua_xep_hang.xlsx',
                        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    # Chuẩn bị base64 ảnh biểu đồ
    plot_option_scores_base64 = get_plot_option_scores_pie_base64()
    plot_criteria_weights_base64 = get_plot_criteria_weights_pie_base64()

    pairwise_matrices = {
        "Ma trận so sánh tiêu chí": criteria_matrix_np.tolist()
    }

    return render_template('step5.html',
                          criteria_cr=criteria_cr,
                          lambda_max=lambda_max_criteria,
                          criteria_ci=(lambda_max_criteria - len(criteria)) / (len(criteria) - 1) if len(criteria) > 1 else 0,
                          criteria=criteria,
                          criteria_weights=criteria_weights,
                          pairwise_matrices=pairwise_matrices,
                          df_results=df_results.to_dict(orient='records'),
                          plot_option_scores_base64=plot_option_scores_base64,
                          plot_criteria_weights_base64=plot_criteria_weights_base64,
                          zip=zip)

@app.route('/plot_criteria_weights.png')
def plot_criteria_weights():
    """Tạo biểu đồ trọng số của các tiêu chí."""
    criteria = session.get('criteria')
    criteria_weights = session.get('criteria_weights')
    
    if not criteria or not criteria_weights:
        return "Không có dữ liệu để tạo biểu đồ trọng số tiêu chí.", 404
    
    plt.figure(figsize=(10, 6)) # Kích thước biểu đồ lớn hơn
    plt.bar(criteria, criteria_weights, color='skyblue')
    plt.title('Trọng số của các tiêu chí', fontsize=16)
    plt.ylabel('Trọng số', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10) # Xoay nhãn trục x để dễ đọc
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout() # Đảm bảo tất cả các thành phần vừa vặn trong hình

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close() # Đóng biểu đồ để giải phóng bộ nhớ
    img.seek(0)

    return Response(img.getvalue(), mimetype='image/png')


@app.route('/plot_alternative_scores.png')
def plot_alternative_scores():
    """Tạo biểu đồ điểm tổng hợp của các phương án."""
    options = session.get('options')
    df_results_dict = session.get('df_results') # Lấy data đã lưu dưới dạng dict

    if not options or not df_results_dict:
        return "Không có dữ liệu để tạo biểu đồ điểm tổng hợp phương án.", 404

    df = pd.DataFrame(df_results_dict)

    plt.figure(figsize=(12, 7)) # Kích thước biểu đồ lớn hơn
    # Sắp xếp lại df theo điểm tổng hợp để biểu đồ có thứ tự
    df_sorted = df.sort_values(by='Điểm tổng hợp', ascending=False)
    plt.bar(df_sorted['Phương án'], df_sorted['Điểm tổng hợp'], color='coral')
    plt.title('Điểm tổng hợp của các phương án', fontsize=16)
    plt.ylabel('Điểm', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10) # Xoay nhãn trục x để dễ đọc
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)

    return Response(img.getvalue(), mimetype='image/png')

# Biến toàn cục lưu lịch sử (LƯU Ý: Không dùng cho môi trường sản xuất đa người dùng)
history_list = []

@app.route('/save_history', methods=['POST'])
def save_history():
    """Lưu mô tả kết quả vào lịch sử."""
    description = request.form.get('description')
    if description:
        history_list.append(description)
        return "Đã lưu lịch sử", 200
    return "Không có dữ liệu để lưu", 400

@app.route('/history')
def history():
    """Hiển thị trang lịch sử."""
    return render_template('history.html', history=history_list)


@app.route('/plot_criteria_weights_pie.png')
def plot_criteria_weights_pie():
    criteria = session.get('criteria')
    criteria_weights = session.get('criteria_weights')
    if not criteria or not criteria_weights:
        print("Error: No criteria or criteria_weights in session")
        return "Không có dữ liệu để tạo biểu đồ trọng số tiêu chí.", 404
    
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.Set2.colors[:len(criteria_weights)]  # Màu Set2 cho tiêu chí
    explode = [0.05] * len(criteria_weights)  # Tách nhẹ các lát
    plt.pie(criteria_weights, labels=criteria, autopct='%1.1f%%', startangle=140, colors=colors, explode=explode, textprops={'fontsize': 12})
    plt.title('Phân Bố Trọng Số Các Tiêu Chí', fontsize=16, pad=20)
    plt.tight_layout()

    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    plt.close(fig)
    img.seek(0)
    buf = BytesIO()
    buf.seek(0)
    response = make_response(buf.read())
    response.headers['Content-Type'] = 'image/png'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return Response(img.getvalue(), mimetype='image/png')

@app.route('/plot_option_scores_pie.png')
def plot_option_scores_pie():
    option_labels = session.get('option_labels')
    option_scores = session.get('option_scores')
    
    if not option_labels or not option_scores:
        return "Không có dữ liệu để tạo biểu đồ điểm tổng hợp phương án.", 404

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.Set3.colors[:len(option_scores)]
    explode = [0.1 if i == 0 else 0 for i in range(len(option_scores))]

    ax.pie(
        option_scores,
        labels=option_labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        explode=explode,
        textprops={'fontsize': 12}
    )
    ax.set_title('Phân Bố Điểm Tổng Hợp Các Phương Án', fontsize=16, pad=20)

    plt.tight_layout()
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    plt.close(fig)
    img.seek(0)

    # Trả về ảnh đúng cách
    response = make_response(img.getvalue())
    response.headers['Content-Type'] = 'image/png'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response
# Chạy ứng dụng Flask

def get_plot_option_scores_pie_base64():
    option_labels = session.get('option_labels')
    option_scores = session.get('option_scores')
    
    if not option_labels or not option_scores:
        return None
    
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.Set3.colors[:len(option_scores)]
    explode = [0.1 if i == 0 else 0 for i in range(len(option_scores))]

    ax.pie(
        option_scores,
        labels=option_labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        explode=explode,
        textprops={'fontsize': 12}
    )
    ax.set_title('Phân Bố Điểm Tổng Hợp Các Phương Án', fontsize=16, pad=20)

    plt.tight_layout()
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    plt.close(fig)
    img.seek(0)

    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

if __name__ == '__main__':
    # Chế độ debug chỉ nên dùng trong phát triển, không dùng trong sản xuất
    app.run(debug=True)
