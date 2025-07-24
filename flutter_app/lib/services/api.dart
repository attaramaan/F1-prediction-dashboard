import 'package:dio/dio.dart';
import '../models/prediction.dart';

class F1Api {
  final Dio _dio;
  F1Api(String baseUrl): _dio = Dio(BaseOptions(baseUrl: baseUrl));

  Future<List<Prediction>> getPrediction(int year, int round, {double wGrid=1.0, double wForm=1.0, double wTeam=1.0}) async {
    final res = await _dio.get('/predict', queryParameters: {
      'year': year,
      'round': round,
      'w_grid': wGrid,
      'w_form': wForm,
      'w_team': wTeam,
    });
    return (res.data as List).map((e)=>Prediction.fromJson(e)).toList();
  }
}
