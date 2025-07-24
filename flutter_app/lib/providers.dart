import 'package:hooks_riverpod/hooks_riverpod.dart';
import 'services/api.dart';

final apiProvider = Provider<F1Api>((ref){
  const baseUrl = 'http://192.168.178.25:8000';
  return F1Api(baseUrl);
});
