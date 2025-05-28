# from flask import Flask, render_template, request, redirect, url_for, send_file, session
# import numpy as np
# import pandas as pd
# from ahp_utils import calculate_weights, consistency_ratio, rank_options
# from io import BytesIO

# app = Flask(__name__)
# app.secret_key = 'your_secret_key'  # Thay bằng key của bạn

# # Danh sách mặc định tiêu chí và phương án (dùng cho chọn ở bước 2)
# AVAILABLE_CRITERIA = [
#     "Chi phí & mức phí bảo hiểm",
#     "Phạm vi bảo hiểm",
#     "Uy tín & đánh giá",
#     "Dịch vụ khách hàng",
#     "Điều khoản hợp đồng"
# ]

# AVAILABLE_OPTIONS = [
#     "Bảo Việt Nhân Thọ",
#     "Prudential Việt Nam",
#     "Dai-ichi Việt Nam",
#     "Hanwha Life Việt Nam",
#     "Manulife Việt Nam"
# ]

# @app.route('/', methods=['GET', 'POST'])
# def step1():
#     error = None
#     if request.method == 'POST':
#         try:
#             n_criteria = int(request.form['n_criteria'])
#             n_options = int(request.form['n_options'])
#             if n_criteria < 4 or n_options < 4:
#                 error = "Phải chọn số lượng tiêu chí và phương án từ 4 trở lên."
#             else:
#                 session['n_criteria'] = n_criteria
#                 session['n_options'] = n_options
#                 return redirect(url_for('step2'))
#         except ValueError:
#             error = "Vui lòng nhập số nguyên hợp lệ."
#     return render_template('step1.html', error=error)


# @app.route('/step2', methods=['GET', 'POST'])
# def step2():
#     error = None
#     n_criteria = session.get('n_criteria', 4)
#     n_options = session.get('n_options', 4)

#     if request.method == 'POST':
#         try:
#             # Lấy tiêu chí và phương án từ form
#             selected_criteria = request.form.getlist('criteria[]')
#             selected_options = request.form.getlist('options[]')

#             # Kiểm tra xem người dùng có chọn đủ không
#             if len(set(selected_criteria)) != n_criteria or len(set(selected_options)) != n_options:
#                 error = "Bạn phải chọn đủ và không được lặp các tiêu chí hoặc phương án."
#             else:
#                 session['criteria'] = selected_criteria
#                 session['options'] = selected_options
#                 return redirect(url_for('step3'))
#         except Exception:
#             error = "Dữ liệu không hợp lệ, vui lòng thử lại."

#     return render_template(
#         'step2.html',
#         error=error,
#         n_criteria=n_criteria,
#         n_options=n_options,
#         available_criteria=AVAILABLE_CRITERIA,
#         available_options=AVAILABLE_OPTIONS,
#         selected_criteria=session.get('criteria', []),
#         selected_options=session.get('options', [])
#     )


# @app.route('/step3', methods=['GET', 'POST'])
# def step3():
#     n_criteria = session.get('n_criteria')
#     criteria = session.get('criteria')  # Lấy tiêu chí từ session

#     if not n_criteria or not criteria:
#         return redirect(url_for('step1'))

#     error = None
#     matrix = None

#     if request.method == 'POST':
#         try:
#             # Xây dựng ma trận tiêu chí từ form
#             matrix = []
#             for i in range(n_criteria):
#                 row = []
#                 for j in range(n_criteria):
#                     val = float(request.form.get(f'criteria_{i}_{j}', 1))
#                     row.append(val)
#                 matrix.append(row)
#             matrix = np.array(matrix)
            
#             # Kiểm tra nhất quán
#             cr = consistency_ratio(matrix)
#             if cr > 0.1:
#                 error = f"Chỉ số nhất quán CR={cr:.3f} > 0.1. Vui lòng nhập lại ma trận."
#             else:
#                 weights, _ = calculate_weights(matrix)
#                 session['criteria_matrix'] = matrix.tolist()
#                 session['criteria_weights'] = weights.tolist()
#                 session['criteria_cr'] = cr
#                 return redirect(url_for('step4'))
#         except Exception:
#             error = "Dữ liệu không hợp lệ, vui lòng nhập lại."

#     return render_template(
#         'step3.html',
#         n_criteria=n_criteria,
#         criteria=criteria,
#         error=error,
#         matrix=matrix
#     )



# @app.route('/step4', methods=['GET', 'POST'])
# def step4():
#     # Lấy dữ liệu từ session
#     n_criteria = session.get('n_criteria')
#     n_options = session.get('n_options')
#     criteria = session.get('criteria')
#     options = session.get('options')
    
#     # Kiểm tra dữ liệu session
#     if not (n_criteria and n_options and criteria and options):
#         return redirect(url_for('step1'))  # Quay lại bước 1 nếu thiếu dữ liệu
    
#     error = None

#     if request.method == 'POST':
#         try:
#             option_matrices = []
#             for c_index in range(n_criteria):
#                 matrix = []
#                 for i in range(n_options):
#                     row = []
#                     for j in range(n_options):
#                         val = float(request.form.get(f'criteria{c_index}_option_{i}_{j}', 1))
#                         row.append(val)
#                     matrix.append(row)
#                 matrix = np.array(matrix)
#                 option_matrices.append(matrix)

#             # Lưu ma trận so sánh vào session
#             session['option_matrices'] = [m.tolist() for m in option_matrices]
#             return redirect(url_for('step5'))  # Chuyển sang bước 5
#         except Exception:
#             error = "Dữ liệu không hợp lệ, vui lòng nhập lại."

#     return render_template(
#         'step4.html',
#         n_criteria=n_criteria,
#         n_options=n_options,
#         criteria=criteria,
#         options=options,
#         error=error,
#         enumerate=enumerate  # Truyền hàm enumerate vào template
#     )




# @app.route('/step5', methods=['GET', 'POST'])
# def step5():
#     n_criteria = session.get('n_criteria')
#     n_options = session.get('n_options')
#     criteria = session.get('criteria')
#     options = session.get('options')
#     criteria_matrix = session.get('criteria_matrix')  # Ma trận so sánh tiêu chí (n_criteria x n_criteria)
#     option_matrices = session.get('option_matrices')  # Danh sách ma trận điểm của các phương án theo từng tiêu chí

#     # Kiểm tra dữ liệu
#     if not all([n_criteria, n_options, criteria, options, criteria_matrix, option_matrices]):
#         return render_template('step5.html', error="Dữ liệu không đầy đủ. Vui lòng kiểm tra lại các bước trước.")

#     try:
#         # Tính chỉ số CR và trọng số tiêu chí
#         criteria_matrix_np = np.array(criteria_matrix)
#         criteria_cr, criteria_weights = calculate_cr(criteria_matrix_np)

#         if criteria_cr > 0.1:
#             return render_template('step5.html', error=f"Chỉ số CR của ma trận tiêu chí vượt ngưỡng cho phép: {criteria_cr:.4f}")

#         # Tính trọng số phương án theo từng tiêu chí
#         option_weights_per_criterion = []
#         for c_index, matrix in enumerate(option_matrices):
#             matrix_np = np.array(matrix)
#             cr, weights = calculate_cr(matrix_np)
#             if cr > 0.1:
#                 return render_template('step5.html', error=f"Chỉ số CR của tiêu chí '{criteria[c_index]}' vượt ngưỡng cho phép.")
#             option_weights_per_criterion.append(weights)

#         # Chuyển danh sách trọng số phương án theo từng tiêu chí thành mảng numpy (n_criteria x n_options)
#         option_weights_matrix = np.vstack(option_weights_per_criterion)

#         # Tính điểm tổng hợp theo trọng số tiêu chí
#         overall_scores = criteria_weights @ option_weights_matrix  # (n_options, )

#         # Tạo DataFrame kết quả, xếp hạng
#         df_results = pd.DataFrame({
#             'Phương án': options,
#             'Điểm tổng hợp': overall_scores
#         }).sort_values(by='Điểm tổng hợp', ascending=False).reset_index(drop=True)
#         df_results['Xếp hạng'] = df_results.index + 1

#         best_option = df_results.iloc[0]['Phương án'] if not df_results.empty else "Không xác định"

#     except Exception as e:
#         return render_template('step5.html', error=f"Lỗi khi tính toán: {e}")

#     # Nếu người dùng yêu cầu tải file excel
#     if request.method == 'POST' and 'download' in request.form:
#         output = io.BytesIO()
#         with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
#             df_results.to_excel(writer, index=False, sheet_name='Kết quả')
#             writer.save()
#         output.seek(0)
#         return send_file(output, as_attachment=True, download_name='ket_qua_xep_hang.xlsx',
#                          mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

#     return render_template('step5.html',
#                            criteria_cr=criteria_cr,
#                            criteria=criteria,
#                            criteria_weights=criteria_weights,
#                            df_results=df_results.to_dict(orient='records'),
#                            best_option=best_option)





# @app.route('/history')
# def history():
#     return "Chức năng lịch sử đang phát triển."


# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, render_template, request, redirect, url_for, send_file, session
import numpy as np
import pandas as pd
from io import BytesIO

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Thay bằng key của bạn

# Danh sách mặc định tiêu chí và phương án (dùng cho chọn ở bước 2)
AVAILABLE_CRITERIA = [
    "Chi phí & mức phí bảo hiểm",
    "Phạm vi bảo hiểm",
    "Uy tín & đánh giá",
    "Dịch vụ khách hàng",
    "Điều khoản hợp đồng"
]

AVAILABLE_OPTIONS = [
    "Bảo Việt Nhân Thọ",
    "Prudential Việt Nam",
    "Dai-ichi Việt Nam",
    "Hanwha Life Việt Nam",
    "Manulife Việt Nam"
]

# Hàm tính trọng số eigenvector và Consistency Ratio (CR)
def calculate_weights(matrix):
    eigvals, eigvecs = np.linalg.eig(matrix)
    max_index = np.argmax(eigvals.real)
    max_eigval = eigvals.real[max_index]
    weights = eigvecs[:, max_index].real
    weights = weights / np.sum(weights)
    return weights, max_eigval

def consistency_ratio(matrix):
    n = matrix.shape[0]
    weights, max_eigval = calculate_weights(matrix)
    ci = (max_eigval - n) / (n - 1)
    # Bảng RI (Random Index) cho n từ 1 đến 10 (theo chuẩn AHP)
    ri_dict = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    ri = ri_dict.get(n, 1.49)  # Nếu n > 10, lấy RI = 1.49
    if ri == 0:
        return 0  # Với n=1 hoặc 2, CR luôn = 0
    cr = ci / ri
    return cr

@app.route('/step1', methods=['GET', 'POST'])
def step1():
    error = None
    if request.method == 'POST':
        try:
            n_criteria = int(request.form['n_criteria'])
            n_options = int(request.form['n_options'])
            if n_criteria < 4 or n_options < 4:
                error = "Phải chọn số lượng tiêu chí và phương án từ 4 trở lên."
            else:
                session['n_criteria'] = n_criteria
                session['n_options'] = n_options
                return redirect(url_for('step2'))
        except ValueError:
            error = "Vui lòng nhập số nguyên hợp lệ."
    return render_template('step1.html', error=error)


@app.route('/step2', methods=['GET', 'POST'])
def step2():
    error = None
    n_criteria = session.get('n_criteria', 4)
    n_options = session.get('n_options', 4)

    if request.method == 'POST':
        try:
            selected_criteria = request.form.getlist('criteria[]')
            selected_options = request.form.getlist('options[]')

            if len(set(selected_criteria)) != n_criteria or len(set(selected_options)) != n_options:
                error = "Bạn phải chọn đủ và không được lặp các tiêu chí hoặc phương án."
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
        selected_criteria=session.get('criteria', []),
        selected_options=session.get('options', [])
    )


@app.route('/step3', methods=['GET', 'POST'])
def step3():
    n_criteria = session.get('n_criteria')
    criteria = session.get('criteria')

    if not n_criteria or not criteria:
        return redirect(url_for('step1'))

    error = None
    matrix = None

    if request.method == 'POST':
        try:
            matrix = []
            for i in range(n_criteria):
                row = []
                for j in range(n_criteria):
                    val = float(request.form.get(f'criteria_{i}_{j}', 1))
                    row.append(val)
                matrix.append(row)
            matrix = np.array(matrix)
            
            cr = consistency_ratio(matrix)
            if cr > 0.1:
                error = f"Chỉ số nhất quán CR={cr:.3f} > 0.1. Vui lòng nhập lại ma trận."
            else:
                weights, _ = calculate_weights(matrix)
                session['criteria_matrix'] = matrix.tolist()
                session['criteria_weights'] = weights.tolist()
                session['criteria_cr'] = cr
                return redirect(url_for('step4'))
        except Exception:
            error = "Dữ liệu không hợp lệ, vui lòng nhập lại."

    return render_template(
        'step3.html',
        n_criteria=n_criteria,
        criteria=criteria,
        error=error,
        matrix=matrix
    )


@app.route('/step4', methods=['GET', 'POST'])
def step4():
    n_criteria = session.get('n_criteria')
    n_options = session.get('n_options')
    criteria = session.get('criteria')
    options = session.get('options')
    
    if not (n_criteria and n_options and criteria and options):
        return redirect(url_for('step1'))
    
    error = None

    if request.method == 'POST':
        try:
            option_matrices = []
            for c_index in range(n_criteria):
                matrix = []
                for i in range(n_options):
                    row = []
                    for j in range(n_options):
                        val = float(request.form.get(f'criteria{c_index}_option_{i}_{j}', 1))
                        row.append(val)
                    matrix.append(row)
                matrix = np.array(matrix)
                option_matrices.append(matrix)

            session['option_matrices'] = [m.tolist() for m in option_matrices]
            return redirect(url_for('step5'))
        except Exception:
            error = "Dữ liệu không hợp lệ, vui lòng nhập lại."

    return render_template(
        'step4.html',
        n_criteria=n_criteria,
        n_options=n_options,
        criteria=criteria,
        options=options,
        error=error,
        enumerate=enumerate
    )



# @app.route('/step5', methods=['GET', 'POST'])
# def step5():
#     n_criteria = session.get('n_criteria')
#     n_options = session.get('n_options')
#     criteria = session.get('criteria')
#     options = session.get('options')
#     criteria_matrix = session.get('criteria_matrix')
#     option_matrices = session.get('option_matrices')

#     empty_pairwise = {}

#     if not all([n_criteria, n_options, criteria, options, criteria_matrix, option_matrices]):
#         return render_template('step5.html', 
#                                error="Dữ liệu không đầy đủ. Vui lòng kiểm tra lại các bước trước.", 
#                                pairwise_matrices=empty_pairwise)

#     try:
#         # Chuyển sang numpy array
#         criteria_matrix_np = np.array(criteria_matrix)
#         criteria_cr = consistency_ratio(criteria_matrix_np)
#         criteria_weights, _ = calculate_weights(criteria_matrix_np)


#         if criteria_cr > 0.1:
#             return render_template('step5.html', 
#                                    error=f"Chỉ số CR của ma trận tiêu chí vượt ngưỡng: {criteria_cr:.4f}",
#                                    pairwise_matrices=empty_pairwise)

#         option_weights_per_criterion = []
#         for c_index, matrix in enumerate(option_matrices):
#             matrix_np = np.array(matrix)
#             cr = consistency_ratio(matrix_np)
#             weights, _ = calculate_weights(matrix_np)
#             if cr > 0.1:
#                 return render_template('step5.html', 
#                                        error=f"Chỉ số CR của tiêu chí '{criteria[c_index]}' vượt ngưỡng.",
#                                        pairwise_matrices=empty_pairwise)
#             option_weights_per_criterion.append(weights)

#         option_weights_matrix = np.vstack(option_weights_per_criterion)  # shape: (n_criteria, n_options)
#         overall_scores = criteria_weights @ option_weights_matrix  # shape: (n_options,)

#         # Tạo DataFrame kết quả và xếp hạng
#         df_results = pd.DataFrame({
#             'Phương án': options,
#             'Điểm tổng hợp': overall_scores
#         }).sort_values(by='Điểm tổng hợp', ascending=False).reset_index(drop=True)
#         df_results['Xếp hạng'] = df_results.index + 1

#         for i, criterion in enumerate(criteria):
#             df_results[criterion] = option_weights_matrix[i, :]

#     except Exception as e:
#         return render_template('step5.html', 
#                                error=f"Lỗi khi tính toán: {e}", 
#                                pairwise_matrices=empty_pairwise)

#     if request.method == 'POST' and 'download' in request.form:
#         output = BytesIO()
#         with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
#             df_results.to_excel(writer, index=False, sheet_name='Kết quả AHP')
#         output.seek(0)
#         return send_file(output,
#                          as_attachment=True,
#                          download_name='ket_qua_xep_hang.xlsx',
#                          mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

#     # Truyền ma trận tiêu chí để template show phần trọng số tiêu chí và ma trận so sánh tiêu chí
#     pairwise_matrices = {criteria[i]: criteria_matrix_np.tolist() for i in range(len(criteria))}

#     return render_template('step5.html',
#                            criteria_cr=criteria_cr,
#                            criteria=criteria,
#                            criteria_weights=criteria_weights,
#                            pairwise_matrices=pairwise_matrices,
#                            df_results=df_results.to_dict(orient='records'),
#                            zip=zip)


@app.route('/step5', methods=['GET', 'POST'])
def step5():
    n_criteria = session.get('n_criteria')
    n_options = session.get('n_options')
    criteria = session.get('criteria')
    options = session.get('options')
    criteria_matrix = session.get('criteria_matrix')
    option_matrices = session.get('option_matrices')

    empty_pairwise = {}

    if not all([n_criteria, n_options, criteria, options, criteria_matrix, option_matrices]):
        return render_template('step5.html', 
                               error="Dữ liệu không đầy đủ. Vui lòng kiểm tra lại các bước trước.", 
                               pairwise_matrices=empty_pairwise)

    try:
        # Chuyển sang numpy array
        criteria_matrix_np = np.array(criteria_matrix)
        criteria_cr = consistency_ratio(criteria_matrix_np)
        criteria_weights, _ = calculate_weights(criteria_matrix_np)

        # Tính Lambda max (λ_max)
        weighted_sum = criteria_matrix_np @ criteria_weights
        lambda_max = np.sum(weighted_sum / criteria_weights) / len(criteria_weights)

        if criteria_cr > 0.1:
            return render_template('step5.html', 
                                   error=f"Chỉ số CR của ma trận tiêu chí vượt ngưỡng: {criteria_cr:.4f}",
                                   pairwise_matrices=empty_pairwise)

        option_weights_per_criterion = []
        for c_index, matrix in enumerate(option_matrices):
            matrix_np = np.array(matrix)
            cr = consistency_ratio(matrix_np)
            weights, _ = calculate_weights(matrix_np)
            if cr > 0.1:
                return render_template('step5.html', 
                                       error=f"Chỉ số CR của tiêu chí '{criteria[c_index]}' vượt ngưỡng.",
                                       pairwise_matrices=empty_pairwise)
            option_weights_per_criterion.append(weights)

        option_weights_matrix = np.vstack(option_weights_per_criterion)  # shape: (n_criteria, n_options)
        overall_scores = criteria_weights @ option_weights_matrix  # shape: (n_options,)

        # Tạo DataFrame kết quả và xếp hạng
        df_results = pd.DataFrame({
            'Phương án': options,
            'Điểm tổng hợp': overall_scores
        }).sort_values(by='Điểm tổng hợp', ascending=False).reset_index(drop=True)
        df_results['Xếp hạng'] = df_results.index + 1

        for i, criterion in enumerate(criteria):
            df_results[criterion] = option_weights_matrix[i, :]

    except Exception as e:
        return render_template('step5.html', 
                               error=f"Lỗi khi tính toán: {e}", 
                               pairwise_matrices=empty_pairwise)

    if request.method == 'POST' and 'download' in request.form:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_results.to_excel(writer, index=False, sheet_name='Kết quả AHP')
        output.seek(0)
        return send_file(output,
                         as_attachment=True,
                         download_name='ket_qua_xep_hang.xlsx',
                         mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    # Truyền ma trận tiêu chí để template show phần trọng số tiêu chí và ma trận so sánh tiêu chí
    pairwise_matrices = {criteria[i]: criteria_matrix_np.tolist() for i in range(len(criteria))}

    return render_template('step5.html',
                           criteria_cr=criteria_cr,
                           lambda_max=lambda_max,
                           criteria=criteria,
                           criteria_weights=criteria_weights,
                           pairwise_matrices=pairwise_matrices,
                           df_results=df_results.to_dict(orient='records'),
                           zip=zip)











@app.route('/')
def home():
    return render_template('home.html')


@app.route('/history')
def history():
    return render_template('history.html')

if __name__ == '__main__':
    app.run(debug=True)
