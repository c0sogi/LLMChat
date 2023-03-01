import 'dart:convert';
import 'package:crypto/crypto.dart';
import 'package:http/http.dart' as http;
import '/src/models/models.dart';

String hashString({required String string, required String secretKey}) {
  var hmacSha256 =
      Hmac(sha256, utf8.encode(secretKey)); // Create a HMAC-SHA256 key
  var digest = hmacSha256.convert(utf8.encode(string)); // Calculate the digest
  return base64.encode(digest.bytes); // Encode the digest to base64
}

String? parseQueryParams(Map<String, String>? params) {
  return params?.entries.map((e) => '${e.key}=${e.value}').join('&');
}

String getCurrentTimestamp({int hourDiff = 0}) {
  DateTime now = DateTime.now();
  String year = now.year.toString().substring(2).padLeft(2, '0');
  String month = now.month.toString().padLeft(2, '0');
  String day = now.day.toString().padLeft(2, '0');
  String hour = now.hour.toString().padLeft(2, '0');
  String minute = now.minute.toString().padLeft(2, '0');
  String second = now.second.toString().padLeft(2, '0');
  return '$year$month$day$hour$minute$second';
}

Future<String> requestFastAPI({
  required String url,
  required Crud crud,
  Map<String, String>? queryParams,
  Map<String, String>? headers,
  Object? body,
  String? secret,
}) async {
  String? parsedQs = parseQueryParams(queryParams?.containsKey("key") ?? false
      ? () {
          queryParams!.addAll({"timestamp": getCurrentTimestamp(hourDiff: 9)});
          return queryParams;
        }()
      : queryParams);
  if (secret != null && parsedQs != null) {
    headers
        ?.addAll({"secret": hashString(string: parsedQs, secretKey: secret)});
  }

  Future<http.Response> Function(
    Uri url, {
    Map<String, String>? headers,
    Object? body,
    Encoding? encoding,
  }) cudMethod;

  switch (crud) {
    case Crud.get:
      return utf8.decode((await http.get(
        Uri.parse("$url?$parsedQs"),
        headers: headers,
      ))
          .bodyBytes);
    case Crud.post:
      cudMethod = http.post;
      break;
    case Crud.put:
      cudMethod = http.put;
      break;
    case Crud.delete:
      cudMethod = http.delete;
      break;
  }
  return utf8.decode((await cudMethod(Uri.parse("$url?$parsedQs"),
          headers: headers, body: body))
      .bodyBytes);
}

Future<String> registerAccount({
  required String host,
  required String registerAccountID,
  required String registerAccountPW,
  required bool isNewUser,
}) async =>
    jsonDecode(
      await requestFastAPI(
        url: "https://$host/api/auth/${isNewUser ? 'register' : 'login'}/email",
        headers: {
          "accept": "application/json",
          "Content-Type": "application/json",
        },
        body: jsonEncode({"email": registerAccountID, "pw": registerAccountPW}),
        crud: Crud.post,
      ),
    )["Authorization"];

Future<Map> registerApiKey({
  required String host,
  required String authorization,
}) async =>
    jsonDecode(
      await requestFastAPI(
        url: "https://$host/api/user/apikeys",
        headers: {
          "accept": "application/json",
          "Content-Type": "application/json",
          "Authorization": authorization
        },
        body: jsonEncode({"user_memo": "테스트용"}),
        crud: Crud.post,
      ),
    );

Future<List> viewAllApiKeys({
  required String host,
  required String authorization,
}) async =>
    jsonDecode(
      await requestFastAPI(
        url: "https://$host/api/user/apikeys",
        headers: {"Authorization": authorization},
        crud: Crud.get,
      ),
    );

Future<Map> services({
  required String host,
  required String authorization,
  required String accessKey,
  required String secretKey,
}) async =>
    jsonDecode(
      await requestFastAPI(
        url: "https://$host/api/services",
        queryParams: {
          "key": accessKey,
        },
        secret: secretKey,
        headers: {"Authorization": authorization},
        crud: Crud.get,
      ),
    );

void main() async {
  // Request JWT User Authorization

  String host = "YOUR_DOMAIN.com";
  String registerAccountID = "TEST_ID";
  String registerAccountPW = "TEST_PW";

  String authorization = await registerAccount(
    host: host,
    registerAccountID: registerAccountID,
    registerAccountPW: registerAccountPW,
    isNewUser: false,
  );
  print("Your Authorization: $authorization");

  Map apiKey = await registerApiKey(host: host, authorization: authorization);
  print("Your API access key: ${apiKey["access_key"]}");
  print("Your API secret key: ${apiKey["secret_key"]}");

  List allApiKeys =
      await viewAllApiKeys(host: host, authorization: authorization);
  print("Your all API access keys: $allApiKeys");

  Map service = await services(
    host: host,
    authorization: authorization,
    accessKey: apiKey["access_key"],
    secretKey: apiKey["secret_key"],
  );
  print("Your Service ID: $service");
}
