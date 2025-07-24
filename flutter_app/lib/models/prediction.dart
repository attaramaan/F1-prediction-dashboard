class Prediction {
  final String drv_abbr;
  final String team;
  final double grid_pos;
  final double win_prob;

  Prediction({required this.drv_abbr, required this.team, required this.grid_pos, required this.win_prob});

  factory Prediction.fromJson(Map<String, dynamic> json){
    return Prediction(
      drv_abbr: json['drv_abbr'] as String,
      team: json['team'] as String,
      grid_pos: (json['grid_pos'] as num).toDouble(),
      win_prob: (json['win_prob'] as num).toDouble(),
    );
  }
}
