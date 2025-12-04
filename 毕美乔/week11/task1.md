# 【作业1】
## 1.数据管理的接口
### 1）上传 PDF 文档
`POST /api/data/upload_pdf`
- 功能描述：  
上传一个 PDF（图文混排），系统会自动进行：
PDF 解析（文本抽取、图像抽取）
embedding 生成
存入向量库

- Request (multipart/form-data)
<div align="center">

| 字段 | 类型 | 描述 |
|:-----:|:-----:|:-----:|
|file| File| PDF文档 |
		
</div> 

- Response

```json
{
  "status": "success",
  "document_id": "doc_202501011233",
  "pages": 87,
  "images_extracted": 35,
  "message": "PDF uploaded and indexed successfully."
}
```
### 2）删除 PDF 文档
`DELETE /api/data/delete/{document_id}`

- Response

```json
{
  "status": "success",
  "document_id": "doc_202501011233",
  "message": "Document and associated embeddings deleted."
}
```

### 3）查询已上传文档列表  
`GET /api/data/list`

- 功能
     1. 列出所有已上传的 PDF 文档
     2. 返回每个文档的多模态解析状态、统计数、完备度
     3. 可以**分页**、**搜索**、**排序**
     4. 方便前端展示、后台监控、RAG 管理

- Query 参数（可选）

<div align="center">
  
| 参数        | 类型       | 说明                                                                   |
| --------- | :--------: | -------------------------------------------------------------------- |
| page      | int      | 页码（默认 1）                                                             |
| page_size | int      | 每页数量（默认 20）                                                          |
| keyword   | string   | 文档名模糊搜索                                                              |
| status    | enum     | 过滤文档解析状态，例如：`uploaded` / `parsed` / `embeddings_generated` / `error` |
| sort_by   | string   | 排序字段（uploaded_at / chunk_count / image_count 等）                      |
| order     | asc/desc | 排序方式                                                                 |

</div> 

- Response

```json

{
  "total": 3,
  "page": 1,
  "page_size": 20,
  "documents": [
    {
      "document_id": "doc_202501011233",
      "title": "艾力斯-公司深度报告商业化成绩显著产品矩阵持续拓宽-25070718页.pdf",
      "pages": 87,
      "file_size": 1423456,
      "uploaded_at": "2025-01-01 10:23:11",
      
      "status": "embeddings_generated",   
      "parse_progress": 100,
      "embedding_progress": 100,

      "statistics": {
        "chunk_count": 312,
        "text_chunk_count": 280,
        "image_count": 48,
        "table_count": 12,
        "multimodal_units": 60
      },

      "error_message": null
    }
  ]
}


```

### 4）获取文档的抽取内容（调试用）
`GET /api/data/{document_id}/content`

- Path 参数

<div align="center">

| 字段 | 类型 | 描述 |
|:-----:|:-----:|:-----:|
|document_id| string| PDF 文档唯一 ID |
		
</div> 

- Query 参数（可选）

<div align="center">

| 字段               | 类型  | 描述                     |
|-------------------|:-----:|--------------------------|
| include_images     | bool  | 是否返回图像信息（默认 true） |
| include_ocr        | bool  | 是否返回 OCR 文本（默认 true） |
| include_nearby_text| bool  | 是否返回邻近文本（默认 true） |

</div>

- 功能描述：
返回指定文档的全部抽取内容，内容已经 按图文 chunk 划分，每个 chunk 包含：
     1. 文本
     2. 关联图像
     3. OCR 文本（图中文字）
     4. 文本周边上下文（nearby text）
     5. chunk 元信息（id、页码、来源位置等）

- Response

```json
{
  "document_id": "doc_202501011233",
  "title": "艾力斯-公司深度报告商业化成绩显著产品矩阵持续拓宽-25070718页",
  "pages": 87,
  "chunks": [
    {
      "chunk_id": "blk_001",
      "page": 12,
      "text": "内容1",
      "images": [
        {
          "image_id": "img_001",
          "image_path": "data/pdf123/images/img_p12_001.png",
          "caption": "趋势图",
          "ocr_text": "colorbar 0–20"
        }
      ],
      "nearby_text": [
        {
          "unit_id": "blk_002",
          "text": "内容2"
        }
      ]
    },
    {
      "chunk_id": "blk_002",
      "page": 12,
      "text": "内容2",
      "images": [
        {
          "image_id": "img_001",
          "image_path": "data/pdf123/images/img_p12_001.png",
          "caption": "趋势图",
          "ocr_text": "colorbar 0–20"
        }
      ],
      "nearby_text": []
    }
  ]
}

```
