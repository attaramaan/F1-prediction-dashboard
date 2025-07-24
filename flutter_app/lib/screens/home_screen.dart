import 'package:flutter/material.dart';
import 'package:flutter_hooks/flutter_hooks.dart';
import 'package:hooks_riverpod/hooks_riverpod.dart';
import '../providers.dart';
import '../models/prediction.dart';

class HomeScreen extends HookConsumerWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final year = useState(2025);
    final round = useState(12);
    final wGrid = useState(1.0);
    final wForm = useState(1.0);
    final wTeam = useState(1.0);
    final preds = useState<List<Prediction>>([]);

    Future<void> load() async {
      final api = ref.read(apiProvider);
      preds.value = await api.getPrediction(year.value, round.value,
          wGrid: wGrid.value, wForm: wForm.value, wTeam: wTeam.value);
    }

    useEffect(() { load(); return null; }, []);

    return Scaffold(
      appBar: AppBar(title: const Text('F1 Winner Predictor')),
      body: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          children: [
            Row(children: [
              Expanded(child: _numField("Year", year)),
              const SizedBox(width: 12),
              Expanded(child: _numField("Round", round)),
            ]),
            const SizedBox(height: 8),
            _slider("Grid weight", wGrid),
            _slider("Driver form weight", wForm),
            _slider("Team form weight", wTeam),
            const SizedBox(height: 8),
            ElevatedButton(onPressed: load, child: const Text("Update Predictions")),
            const SizedBox(height: 12),
            if (preds.value.isNotEmpty) _podium(preds.value),
            const SizedBox(height: 12),
            Expanded(child: _list(preds.value)),
          ],
        ),
      ),
    );
  }

  Widget _numField(String label, ValueNotifier<int> state){
    return TextFormField(
      initialValue: state.value.toString(),
      keyboardType: TextInputType.number,
      decoration: InputDecoration(labelText: label),
      onFieldSubmitted: (v)=> state.value = int.tryParse(v) ?? state.value,
    );
  }

  Widget _slider(String label, ValueNotifier<double> state){
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Text(label),
      Slider(value: state.value, min: 0.1, max: 3.0, divisions: 29,
        label: state.value.toStringAsFixed(1),
        onChanged: (v)=> state.value = v)
    ]);
  }

  Widget _podium(List<Prediction> list){
    final pod = list.take(3).toList();
    return Row(mainAxisAlignment: MainAxisAlignment.spaceEvenly, children: [
      if (pod.length>=2) _card(pod[1], top:false),
      if (pod.isNotEmpty) _card(pod[0], top:true),
      if (pod.length>=3) _card(pod[2], top:false),
    ]);
  }

  Widget _card(Prediction p,{required bool top}){
    return Container(
      width: 110,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: top? Colors.amber.shade100: Colors.grey.shade200,
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(children: [
        Text(p.drv_abbr, style: const TextStyle(fontSize: 22,fontWeight: FontWeight.bold)),
        Text(p.team, style: const TextStyle(fontSize: 12,color: Colors.black54)),
        const SizedBox(height: 6),
        Text('${(p.win_prob*100).toStringAsFixed(1)} %', style: TextStyle(color: Colors.green.shade700,fontWeight: FontWeight.w600)),
      ]),
    );
  }

  Widget _list(List<Prediction> list){
    return ListView.separated(
      itemCount: list.length,
      separatorBuilder: (_, __)=> const Divider(height:1),
      itemBuilder: (_, i){
        final p = list[i];
        return ListTile(
          leading: Text(p.drv_abbr, style: const TextStyle(fontWeight: FontWeight.bold)),
          title: Text(p.team),
          subtitle: Text('Grid ${p.grid_pos.toInt()}'),
          trailing: Text('${(p.win_prob*100).toStringAsFixed(1)} %'),
        );
      },
    );
  }
}
