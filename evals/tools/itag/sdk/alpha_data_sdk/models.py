# -*- coding: utf-8 -*-
# This file is auto-generated, don't edit it. Thanks.
from Tea.model import TeaModel
from typing import Dict, Any, List, BinaryIO


class PeriodConfig(TeaModel):
    def __init__(
        self,
        type: str = None,
        time_mode: str = None,
        assign_times: str = None,
        begin: str = None,
        end: str = None,
        interval_hour: str = None,
        runtime: str = None,
        depend_node: str = None,
    ):
        self.type = type
        self.time_mode = time_mode
        self.assign_times = assign_times
        self.begin = begin
        self.end = end
        self.interval_hour = interval_hour
        self.runtime = runtime
        self.depend_node = depend_node

    def validate(self):
        pass

    def to_map(self):
        _map = super().to_map()
        if _map is not None:
            return _map

        result = dict()
        if self.type is not None:
            result['type'] = self.type
        if self.time_mode is not None:
            result['timeMode'] = self.time_mode
        if self.assign_times is not None:
            result['assignTimes'] = self.assign_times
        if self.begin is not None:
            result['begin'] = self.begin
        if self.end is not None:
            result['end'] = self.end
        if self.interval_hour is not None:
            result['intervalHour'] = self.interval_hour
        if self.runtime is not None:
            result['runtime'] = self.runtime
        if self.depend_node is not None:
            result['dependNode'] = self.depend_node
        return result

    def from_map(self, m: dict = None):
        m = m or dict()
        if m.get('type') is not None:
            self.type = m.get('type')
        if m.get('timeMode') is not None:
            self.time_mode = m.get('timeMode')
        if m.get('assignTimes') is not None:
            self.assign_times = m.get('assignTimes')
        if m.get('begin') is not None:
            self.begin = m.get('begin')
        if m.get('end') is not None:
            self.end = m.get('end')
        if m.get('intervalHour') is not None:
            self.interval_hour = m.get('intervalHour')
        if m.get('runtime') is not None:
            self.runtime = m.get('runtime')
        if m.get('dependNode') is not None:
            self.depend_node = m.get('dependNode')
        return self


class UserInfo(TeaModel):
    def __init__(
        self,
        id: str = None,
        channel: str = None,
        work_no: str = None,
        nick_name: str = None,
        tnt_inst_id: str = None,
        alpha_user_id: int = None,
        domain: str = None,
        alpha_user_tnt_inst_map: Dict[str, str] = None,
        operator_name: str = None,
        auth_type: str = None,
        token_create_time: str = None,
    ):
        self.id = id
        self.channel = channel
        self.work_no = work_no
        self.nick_name = nick_name
        self.tnt_inst_id = tnt_inst_id
        self.alpha_user_id = alpha_user_id
        self.domain = domain
        self.alpha_user_tnt_inst_map = alpha_user_tnt_inst_map
        self.operator_name = operator_name
        self.auth_type = auth_type
        self.token_create_time = token_create_time

    def validate(self):
        pass

    def to_map(self):
        _map = super().to_map()
        if _map is not None:
            return _map

        result = dict()
        if self.id is not None:
            result['id'] = self.id
        if self.channel is not None:
            result['channel'] = self.channel
        if self.work_no is not None:
            result['workNo'] = self.work_no
        if self.nick_name is not None:
            result['nickName'] = self.nick_name
        if self.tnt_inst_id is not None:
            result['tntInstId'] = self.tnt_inst_id
        if self.alpha_user_id is not None:
            result['alphaUserId'] = self.alpha_user_id
        if self.domain is not None:
            result['domain'] = self.domain
        if self.alpha_user_tnt_inst_map is not None:
            result['alphaUserTntInstMap'] = self.alpha_user_tnt_inst_map
        if self.operator_name is not None:
            result['operatorName'] = self.operator_name
        if self.auth_type is not None:
            result['authType'] = self.auth_type
        if self.token_create_time is not None:
            result['tokenCreateTime'] = self.token_create_time
        return result

    def from_map(self, m: dict = None):
        m = m or dict()
        if m.get('id') is not None:
            self.id = m.get('id')
        if m.get('channel') is not None:
            self.channel = m.get('channel')
        if m.get('workNo') is not None:
            self.work_no = m.get('workNo')
        if m.get('nickName') is not None:
            self.nick_name = m.get('nickName')
        if m.get('tntInstId') is not None:
            self.tnt_inst_id = m.get('tntInstId')
        if m.get('alphaUserId') is not None:
            self.alpha_user_id = m.get('alphaUserId')
        if m.get('domain') is not None:
            self.domain = m.get('domain')
        if m.get('alphaUserTntInstMap') is not None:
            self.alpha_user_tnt_inst_map = m.get('alphaUserTntInstMap')
        if m.get('operatorName') is not None:
            self.operator_name = m.get('operatorName')
        if m.get('authType') is not None:
            self.auth_type = m.get('authType')
        if m.get('tokenCreateTime') is not None:
            self.token_create_time = m.get('tokenCreateTime')
        return self


class DatasetExifSourceVisitInfoAccessInfo(TeaModel):
    def __init__(
        self,
        access_key_id: str = None,
        secret_access_key: str = None,
        bucket: str = None,
        source_import_type: str = None,
        endpoint: str = None,
        filepath: str = None,
    ):
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.bucket = bucket
        self.source_import_type = source_import_type
        self.endpoint = endpoint
        self.filepath = filepath

    def validate(self):
        pass

    def to_map(self):
        _map = super().to_map()
        if _map is not None:
            return _map

        result = dict()
        if self.access_key_id is not None:
            result['accessKeyId'] = self.access_key_id
        if self.secret_access_key is not None:
            result['secretAccessKey'] = self.secret_access_key
        if self.bucket is not None:
            result['bucket'] = self.bucket
        if self.source_import_type is not None:
            result['source_import_type'] = self.source_import_type
        if self.endpoint is not None:
            result['endpoint'] = self.endpoint
        if self.filepath is not None:
            result['filepath'] = self.filepath
        return result

    def from_map(self, m: dict = None):
        m = m or dict()
        if m.get('accessKeyId') is not None:
            self.access_key_id = m.get('accessKeyId')
        if m.get('secretAccessKey') is not None:
            self.secret_access_key = m.get('secretAccessKey')
        if m.get('bucket') is not None:
            self.bucket = m.get('bucket')
        if m.get('source_import_type') is not None:
            self.source_import_type = m.get('source_import_type')
        if m.get('endpoint') is not None:
            self.endpoint = m.get('endpoint')
        if m.get('filepath') is not None:
            self.filepath = m.get('filepath')
        return self


class DatasetExifSourceVisitInfo(TeaModel):
    def __init__(
        self,
        type: str = None,
        project: str = None,
        storage: str = None,
        access_info: DatasetExifSourceVisitInfoAccessInfo = None,
        new_store_name: str = None,
        total_num: int = None,
        version: str = None,
        field_schema: str = None,
        storage_format: str = None,
        storage_type: str = None,
        storage_define: str = None,
    ):
        self.type = type
        self.project = project
        self.storage = storage
        self.access_info = access_info
        self.new_store_name = new_store_name
        self.total_num = total_num
        self.version = version
        self.field_schema = field_schema
        self.storage_format = storage_format
        self.storage_type = storage_type
        self.storage_define = storage_define

    def validate(self):
        if self.access_info:
            self.access_info.validate()

    def to_map(self):
        _map = super().to_map()
        if _map is not None:
            return _map

        result = dict()
        if self.type is not None:
            result['type'] = self.type
        if self.project is not None:
            result['project'] = self.project
        if self.storage is not None:
            result['storage'] = self.storage
        if self.access_info is not None:
            result['accessInfo'] = self.access_info.to_map()
        if self.new_store_name is not None:
            result['newStoreName'] = self.new_store_name
        if self.total_num is not None:
            result['totalNum'] = self.total_num
        if self.version is not None:
            result['version'] = self.version
        if self.field_schema is not None:
            result['fieldSchema'] = self.field_schema
        if self.storage_format is not None:
            result['storageFormat'] = self.storage_format
        if self.storage_type is not None:
            result['storageType'] = self.storage_type
        if self.storage_define is not None:
            result['storageDefine'] = self.storage_define
        return result

    def from_map(self, m: dict = None):
        m = m or dict()
        if m.get('type') is not None:
            self.type = m.get('type')
        if m.get('project') is not None:
            self.project = m.get('project')
        if m.get('storage') is not None:
            self.storage = m.get('storage')
        if m.get('accessInfo') is not None:
            temp_model = DatasetExifSourceVisitInfoAccessInfo()
            self.access_info = temp_model.from_map(m['accessInfo'])
        if m.get('newStoreName') is not None:
            self.new_store_name = m.get('newStoreName')
        if m.get('totalNum') is not None:
            self.total_num = m.get('totalNum')
        if m.get('version') is not None:
            self.version = m.get('version')
        if m.get('fieldSchema') is not None:
            self.field_schema = m.get('fieldSchema')
        if m.get('storageFormat') is not None:
            self.storage_format = m.get('storageFormat')
        if m.get('storageType') is not None:
            self.storage_type = m.get('storageType')
        if m.get('storageDefine') is not None:
            self.storage_define = m.get('storageDefine')
        return self


class DatasetExif(TeaModel):
    def __init__(
        self,
        source_visit_info: DatasetExifSourceVisitInfo = None,
        period_config: Dict[str, Any] = None,
    ):
        self.source_visit_info = source_visit_info
        self.period_config = period_config

    def validate(self):
        if self.source_visit_info:
            self.source_visit_info.validate()

    def to_map(self):
        _map = super().to_map()
        if _map is not None:
            return _map

        result = dict()
        if self.source_visit_info is not None:
            result['sourceVisitInfo'] = self.source_visit_info.to_map()
        if self.period_config is not None:
            result['periodConfig'] = self.period_config
        return result

    def from_map(self, m: dict = None):
        m = m or dict()
        if m.get('sourceVisitInfo') is not None:
            temp_model = DatasetExifSourceVisitInfo()
            self.source_visit_info = temp_model.from_map(m['sourceVisitInfo'])
        if m.get('periodConfig') is not None:
            self.period_config = m.get('periodConfig')
        return self


class Dataset(TeaModel):
    def __init__(
        self,
        id: int = None,
        gmt_create: str = None,
        gmt_modified: str = None,
        creator_id: str = None,
        creator_name: str = None,
        modifier_id: str = None,
        modifier_name: str = None,
        name: str = None,
        set_id: int = None,
        set_type: str = None,
        set_type_str: str = None,
        period_info: Dict[str, str] = None,
        period_config: PeriodConfig = None,
        status: str = None,
        shared: bool = None,
        shared_mode: str = None,
        standard: bool = None,
        mount: int = None,
        data_classify: int = None,
        data_source: str = None,
        data_classify_str: str = None,
        data_count: int = None,
        mount_meta_info: str = None,
        algo_type_enum_list: List[str] = None,
        remark: str = None,
        tags: List[str] = None,
        deleted: int = None,
        tnt_inst_id: str = None,
        out_id: str = None,
        owner_list: List[UserInfo] = None,
        partner_list: List[str] = None,
        exif: DatasetExif = None,
        visit_info: Dict[str, Any] = None,
        up_collections: List[str] = None,
        down_collections: List[str] = None,
        msg: str = None,
        mode: str = None,
        biz_type: str = None,
        secure_level: str = None,
        parent_collection_id: str = None,
    ):
        self.id = id
        self.gmt_create = gmt_create
        self.gmt_modified = gmt_modified
        self.creator_id = creator_id
        self.creator_name = creator_name
        self.modifier_id = modifier_id
        self.modifier_name = modifier_name
        self.name = name
        self.set_id = set_id
        self.set_type = set_type
        self.set_type_str = set_type_str
        self.period_info = period_info
        self.period_config = period_config
        self.status = status
        self.shared = shared
        self.shared_mode = shared_mode
        self.standard = standard
        self.mount = mount
        self.data_classify = data_classify
        self.data_source = data_source
        self.data_classify_str = data_classify_str
        self.data_count = data_count
        self.mount_meta_info = mount_meta_info
        self.algo_type_enum_list = algo_type_enum_list
        self.remark = remark
        self.tags = tags
        self.deleted = deleted
        self.tnt_inst_id = tnt_inst_id
        self.out_id = out_id
        self.owner_list = owner_list
        self.partner_list = partner_list
        self.exif = exif
        self.visit_info = visit_info
        self.up_collections = up_collections
        self.down_collections = down_collections
        self.msg = msg
        self.mode = mode
        self.biz_type = biz_type
        self.secure_level = secure_level
        self.parent_collection_id = parent_collection_id

    def validate(self):
        if self.period_config:
            self.period_config.validate()
        if self.owner_list:
            for k in self.owner_list:
                if k:
                    k.validate()
        if self.exif:
            self.exif.validate()

    def to_map(self):
        _map = super().to_map()
        if _map is not None:
            return _map

        result = dict()
        if self.id is not None:
            result['id'] = self.id
        if self.gmt_create is not None:
            result['gmtCreate'] = self.gmt_create
        if self.gmt_modified is not None:
            result['gmtModified'] = self.gmt_modified
        if self.creator_id is not None:
            result['creatorId'] = self.creator_id
        if self.creator_name is not None:
            result['creatorName'] = self.creator_name
        if self.modifier_id is not None:
            result['modifierId'] = self.modifier_id
        if self.modifier_name is not None:
            result['modifierName'] = self.modifier_name
        if self.name is not None:
            result['name'] = self.name
        if self.set_id is not None:
            result['setId'] = self.set_id
        if self.set_type is not None:
            result['setType'] = self.set_type
        if self.set_type_str is not None:
            result['setTypeStr'] = self.set_type_str
        if self.period_info is not None:
            result['periodInfo'] = self.period_info
        if self.period_config is not None:
            result['periodConfig'] = self.period_config.to_map()
        if self.status is not None:
            result['status'] = self.status
        if self.shared is not None:
            result['shared'] = self.shared
        if self.shared_mode is not None:
            result['sharedMode'] = self.shared_mode
        if self.standard is not None:
            result['standard'] = self.standard
        if self.mount is not None:
            result['mount'] = self.mount
        if self.data_classify is not None:
            result['dataClassify'] = self.data_classify
        if self.data_source is not None:
            result['dataSource'] = self.data_source
        if self.data_classify_str is not None:
            result['dataClassifyStr'] = self.data_classify_str
        if self.data_count is not None:
            result['dataCount'] = self.data_count
        if self.mount_meta_info is not None:
            result['mountMetaInfo'] = self.mount_meta_info
        if self.algo_type_enum_list is not None:
            result['algoTypeEnumList'] = self.algo_type_enum_list
        if self.remark is not None:
            result['remark'] = self.remark
        if self.tags is not None:
            result['tags'] = self.tags
        if self.deleted is not None:
            result['deleted'] = self.deleted
        if self.tnt_inst_id is not None:
            result['tntInstId'] = self.tnt_inst_id
        if self.out_id is not None:
            result['outId'] = self.out_id
        result['ownerList'] = []
        if self.owner_list is not None:
            for k in self.owner_list:
                result['ownerList'].append(k.to_map() if k else None)
        if self.partner_list is not None:
            result['partnerList'] = self.partner_list
        if self.exif is not None:
            result['exif'] = self.exif.to_map()
        if self.visit_info is not None:
            result['visitInfo'] = self.visit_info
        if self.up_collections is not None:
            result['upCollections'] = self.up_collections
        if self.down_collections is not None:
            result['downCollections'] = self.down_collections
        if self.msg is not None:
            result['msg'] = self.msg
        if self.mode is not None:
            result['mode'] = self.mode
        if self.biz_type is not None:
            result['bizType'] = self.biz_type
        if self.secure_level is not None:
            result['secureLevel'] = self.secure_level
        if self.parent_collection_id is not None:
            result['parentCollectionId'] = self.parent_collection_id
        return result

    def from_map(self, m: dict = None):
        m = m or dict()
        if m.get('id') is not None:
            self.id = m.get('id')
        if m.get('gmtCreate') is not None:
            self.gmt_create = m.get('gmtCreate')
        if m.get('gmtModified') is not None:
            self.gmt_modified = m.get('gmtModified')
        if m.get('creatorId') is not None:
            self.creator_id = m.get('creatorId')
        if m.get('creatorName') is not None:
            self.creator_name = m.get('creatorName')
        if m.get('modifierId') is not None:
            self.modifier_id = m.get('modifierId')
        if m.get('modifierName') is not None:
            self.modifier_name = m.get('modifierName')
        if m.get('name') is not None:
            self.name = m.get('name')
        if m.get('setId') is not None:
            self.set_id = m.get('setId')
        if m.get('setType') is not None:
            self.set_type = m.get('setType')
        if m.get('setTypeStr') is not None:
            self.set_type_str = m.get('setTypeStr')
        if m.get('periodInfo') is not None:
            self.period_info = m.get('periodInfo')
        if m.get('periodConfig') is not None:
            temp_model = PeriodConfig()
            self.period_config = temp_model.from_map(m['periodConfig'])
        if m.get('status') is not None:
            self.status = m.get('status')
        if m.get('shared') is not None:
            self.shared = m.get('shared')
        if m.get('sharedMode') is not None:
            self.shared_mode = m.get('sharedMode')
        if m.get('standard') is not None:
            self.standard = m.get('standard')
        if m.get('mount') is not None:
            self.mount = m.get('mount')
        if m.get('dataClassify') is not None:
            self.data_classify = m.get('dataClassify')
        if m.get('dataSource') is not None:
            self.data_source = m.get('dataSource')
        if m.get('dataClassifyStr') is not None:
            self.data_classify_str = m.get('dataClassifyStr')
        if m.get('dataCount') is not None:
            self.data_count = m.get('dataCount')
        if m.get('mountMetaInfo') is not None:
            self.mount_meta_info = m.get('mountMetaInfo')
        if m.get('algoTypeEnumList') is not None:
            self.algo_type_enum_list = m.get('algoTypeEnumList')
        if m.get('remark') is not None:
            self.remark = m.get('remark')
        if m.get('tags') is not None:
            self.tags = m.get('tags')
        if m.get('deleted') is not None:
            self.deleted = m.get('deleted')
        if m.get('tntInstId') is not None:
            self.tnt_inst_id = m.get('tntInstId')
        if m.get('outId') is not None:
            self.out_id = m.get('outId')
        self.owner_list = []
        if m.get('ownerList') is not None:
            for k in m.get('ownerList'):
                temp_model = UserInfo()
                self.owner_list.append(temp_model.from_map(k))
        if m.get('partnerList') is not None:
            self.partner_list = m.get('partnerList')
        if m.get('exif') is not None:
            temp_model = DatasetExif()
            self.exif = temp_model.from_map(m['exif'])
        if m.get('visitInfo') is not None:
            self.visit_info = m.get('visitInfo')
        if m.get('upCollections') is not None:
            self.up_collections = m.get('upCollections')
        if m.get('downCollections') is not None:
            self.down_collections = m.get('downCollections')
        if m.get('msg') is not None:
            self.msg = m.get('msg')
        if m.get('mode') is not None:
            self.mode = m.get('mode')
        if m.get('bizType') is not None:
            self.biz_type = m.get('bizType')
        if m.get('secureLevel') is not None:
            self.secure_level = m.get('secureLevel')
        if m.get('parentCollectionId') is not None:
            self.parent_collection_id = m.get('parentCollectionId')
        return self


class DatasetsResults(TeaModel):
    def __init__(
        self,
        result: List[Dataset] = None,
        page_num: int = None,
        page_size: int = None,
        total: int = None,
        total_page: int = None,
    ):
        self.result = result
        self.page_num = page_num
        self.page_size = page_size
        self.total = total
        self.total_page = total_page

    def validate(self):
        if self.result:
            for k in self.result:
                if k:
                    k.validate()

    def to_map(self):
        _map = super().to_map()
        if _map is not None:
            return _map

        result = dict()
        result['result'] = []
        if self.result is not None:
            for k in self.result:
                result['result'].append(k.to_map() if k else None)
        if self.page_num is not None:
            result['pageNum'] = self.page_num
        if self.page_size is not None:
            result['pageSize'] = self.page_size
        if self.total is not None:
            result['total'] = self.total
        if self.total_page is not None:
            result['totalPage'] = self.total_page
        return result

    def from_map(self, m: dict = None):
        m = m or dict()
        self.result = []
        if m.get('result') is not None:
            for k in m.get('result'):
                temp_model = Dataset()
                self.result.append(temp_model.from_map(k))
        if m.get('pageNum') is not None:
            self.page_num = m.get('pageNum')
        if m.get('pageSize') is not None:
            self.page_size = m.get('pageSize')
        if m.get('total') is not None:
            self.total = m.get('total')
        if m.get('totalPage') is not None:
            self.total_page = m.get('totalPage')
        return self


class ListDatasetsRequest(TeaModel):
    def __init__(
        self,
        source: str = None,
        collection_mix_info: str = None,
        name: str = None,
        page_num: int = None,
        page_size: int = None,
        set_type: str = None,
        shared_type: str = None,
        tag: str = None,
        contain_deleted: bool = None,
        creator_id: str = None,
    ):
        self.source = source
        self.collection_mix_info = collection_mix_info
        self.name = name
        self.page_num = page_num
        self.page_size = page_size
        self.set_type = set_type
        self.shared_type = shared_type
        self.tag = tag
        self.contain_deleted = contain_deleted
        self.creator_id = creator_id

    def validate(self):
        pass

    def to_map(self):
        _map = super().to_map()
        if _map is not None:
            return _map

        result = dict()
        if self.source is not None:
            result['source'] = self.source
        if self.collection_mix_info is not None:
            result['collectionMixInfo'] = self.collection_mix_info
        if self.name is not None:
            result['name'] = self.name
        if self.page_num is not None:
            result['pageNum'] = self.page_num
        if self.page_size is not None:
            result['pageSize'] = self.page_size
        if self.set_type is not None:
            result['setType'] = self.set_type
        if self.shared_type is not None:
            result['sharedType'] = self.shared_type
        if self.tag is not None:
            result['tag'] = self.tag
        if self.contain_deleted is not None:
            result['containDeleted'] = self.contain_deleted
        if self.creator_id is not None:
            result['creatorId'] = self.creator_id
        return result

    def from_map(self, m: dict = None):
        m = m or dict()
        if m.get('source') is not None:
            self.source = m.get('source')
        if m.get('collectionMixInfo') is not None:
            self.collection_mix_info = m.get('collectionMixInfo')
        if m.get('name') is not None:
            self.name = m.get('name')
        if m.get('pageNum') is not None:
            self.page_num = m.get('pageNum')
        if m.get('pageSize') is not None:
            self.page_size = m.get('pageSize')
        if m.get('setType') is not None:
            self.set_type = m.get('setType')
        if m.get('sharedType') is not None:
            self.shared_type = m.get('sharedType')
        if m.get('tag') is not None:
            self.tag = m.get('tag')
        if m.get('containDeleted') is not None:
            self.contain_deleted = m.get('containDeleted')
        if m.get('creatorId') is not None:
            self.creator_id = m.get('creatorId')
        return self


class ListDatasetsBody(TeaModel):
    def __init__(
        self,
        code: int = None,
        time: int = None,
        desc: str = None,
        err_msg: str = None,
        succ: bool = None,
        host: str = None,
        trace_id: str = None,
        result: DatasetsResults = None,
    ):
        self.code = code
        self.time = time
        self.desc = desc
        self.err_msg = err_msg
        self.succ = succ
        self.host = host
        self.trace_id = trace_id
        self.result = result

    def validate(self):
        if self.result:
            self.result.validate()

    def to_map(self):
        _map = super().to_map()
        if _map is not None:
            return _map

        result = dict()
        if self.code is not None:
            result['code'] = self.code
        if self.time is not None:
            result['time'] = self.time
        if self.desc is not None:
            result['desc'] = self.desc
        if self.err_msg is not None:
            result['errMsg'] = self.err_msg
        if self.succ is not None:
            result['succ'] = self.succ
        if self.host is not None:
            result['host'] = self.host
        if self.trace_id is not None:
            result['traceId'] = self.trace_id
        if self.result is not None:
            result['result'] = self.result.to_map()
        return result

    def from_map(self, m: dict = None):
        m = m or dict()
        if m.get('code') is not None:
            self.code = m.get('code')
        if m.get('time') is not None:
            self.time = m.get('time')
        if m.get('desc') is not None:
            self.desc = m.get('desc')
        if m.get('errMsg') is not None:
            self.err_msg = m.get('errMsg')
        if m.get('succ') is not None:
            self.succ = m.get('succ')
        if m.get('host') is not None:
            self.host = m.get('host')
        if m.get('traceId') is not None:
            self.trace_id = m.get('traceId')
        if m.get('result') is not None:
            temp_model = DatasetsResults()
            self.result = temp_model.from_map(m['result'])
        return self


class ListDatasetsResponse(TeaModel):
    def __init__(
        self,
        headers: Dict[str, str] = None,
        status_code: int = None,
        body: ListDatasetsBody = None,
    ):
        self.headers = headers
        self.status_code = status_code
        self.body = body

    def validate(self):
        self.validate_required(self.headers, 'headers')
        self.validate_required(self.status_code, 'status_code')
        self.validate_required(self.body, 'body')
        if self.body:
            self.body.validate()

    def to_map(self):
        _map = super().to_map()
        if _map is not None:
            return _map

        result = dict()
        if self.headers is not None:
            result['headers'] = self.headers
        if self.status_code is not None:
            result['statusCode'] = self.status_code
        if self.body is not None:
            result['body'] = self.body.to_map()
        return result

    def from_map(self, m: dict = None):
        m = m or dict()
        if m.get('headers') is not None:
            self.headers = m.get('headers')
        if m.get('statusCode') is not None:
            self.status_code = m.get('statusCode')
        if m.get('body') is not None:
            temp_model = ListDatasetsBody()
            self.body = temp_model.from_map(m['body'])
        return self


class GetDatasetBody(TeaModel):
    def __init__(
        self,
        code: int = None,
        desc: str = None,
        err_msg: str = None,
        succ: bool = None,
        host: str = None,
        trace_id: str = None,
        time: int = None,
        result: Dataset = None,
    ):
        self.code = code
        self.desc = desc
        self.err_msg = err_msg
        self.succ = succ
        self.host = host
        self.trace_id = trace_id
        self.time = time
        self.result = result

    def validate(self):
        if self.result:
            self.result.validate()

    def to_map(self):
        _map = super().to_map()
        if _map is not None:
            return _map

        result = dict()
        if self.code is not None:
            result['code'] = self.code
        if self.desc is not None:
            result['desc'] = self.desc
        if self.err_msg is not None:
            result['errMsg'] = self.err_msg
        if self.succ is not None:
            result['succ'] = self.succ
        if self.host is not None:
            result['host'] = self.host
        if self.trace_id is not None:
            result['traceId'] = self.trace_id
        if self.time is not None:
            result['time'] = self.time
        if self.result is not None:
            result['result'] = self.result.to_map()
        return result

    def from_map(self, m: dict = None):
        m = m or dict()
        if m.get('code') is not None:
            self.code = m.get('code')
        if m.get('desc') is not None:
            self.desc = m.get('desc')
        if m.get('errMsg') is not None:
            self.err_msg = m.get('errMsg')
        if m.get('succ') is not None:
            self.succ = m.get('succ')
        if m.get('host') is not None:
            self.host = m.get('host')
        if m.get('traceId') is not None:
            self.trace_id = m.get('traceId')
        if m.get('time') is not None:
            self.time = m.get('time')
        if m.get('result') is not None:
            temp_model = Dataset()
            self.result = temp_model.from_map(m['result'])
        return self


class GetDatasetResponse(TeaModel):
    def __init__(
        self,
        headers: Dict[str, str] = None,
        status_code: int = None,
        body: GetDatasetBody = None,
    ):
        self.headers = headers
        self.status_code = status_code
        self.body = body

    def validate(self):
        self.validate_required(self.headers, 'headers')
        self.validate_required(self.status_code, 'status_code')
        self.validate_required(self.body, 'body')
        if self.body:
            self.body.validate()

    def to_map(self):
        _map = super().to_map()
        if _map is not None:
            return _map

        result = dict()
        if self.headers is not None:
            result['headers'] = self.headers
        if self.status_code is not None:
            result['statusCode'] = self.status_code
        if self.body is not None:
            result['body'] = self.body.to_map()
        return result

    def from_map(self, m: dict = None):
        m = m or dict()
        if m.get('headers') is not None:
            self.headers = m.get('headers')
        if m.get('statusCode') is not None:
            self.status_code = m.get('statusCode')
        if m.get('body') is not None:
            temp_model = GetDatasetBody()
            self.body = temp_model.from_map(m['body'])
        return self


class OssConf(TeaModel):
    def __init__(
        self,
        key: str = None,
        token: str = None,
        bucket: str = None,
        endpoint: str = None,
        path: str = None,
    ):
        self.key = key
        self.token = token
        self.bucket = bucket
        self.endpoint = endpoint
        self.path = path

    def validate(self):
        pass

    def to_map(self):
        _map = super().to_map()
        if _map is not None:
            return _map

        result = dict()
        if self.key is not None:
            result['key'] = self.key
        if self.token is not None:
            result['token'] = self.token
        if self.bucket is not None:
            result['bucket'] = self.bucket
        if self.endpoint is not None:
            result['endpoint'] = self.endpoint
        if self.path is not None:
            result['path'] = self.path
        return result

    def from_map(self, m: dict = None):
        m = m or dict()
        if m.get('key') is not None:
            self.key = m.get('key')
        if m.get('token') is not None:
            self.token = m.get('token')
        if m.get('bucket') is not None:
            self.bucket = m.get('bucket')
        if m.get('endpoint') is not None:
            self.endpoint = m.get('endpoint')
        if m.get('path') is not None:
            self.path = m.get('path')
        return self


class FieldInfo(TeaModel):
    def __init__(
        self,
        id: str = None,
        gmt_create: str = None,
        gmt_modified: str = None,
        creator_id: str = None,
        modifier_id: str = None,
        field_name: str = None,
        type: str = None,
        field_type: str = None,
        classify: str = None,
        field_classify: str = None,
        category: str = None,
        alias: Dict[str, str] = None,
        nullable: bool = None,
        field_sensitive: bool = None,
        defval: str = None,
        field_primary: bool = None,
        field_desc: str = None,
        biz_tag: bool = None,
        domain: str = None,
        domain_type: str = None,
        collection_id: str = None,
        data_key: str = None,
        field_order: str = None,
        visit_info: str = None,
    ):
        self.id = id
        self.gmt_create = gmt_create
        self.gmt_modified = gmt_modified
        self.creator_id = creator_id
        self.modifier_id = modifier_id
        self.field_name = field_name
        self.type = type
        self.field_type = field_type
        self.classify = classify
        self.field_classify = field_classify
        self.category = category
        self.alias = alias
        self.nullable = nullable
        self.field_sensitive = field_sensitive
        self.defval = defval
        self.field_primary = field_primary
        self.field_desc = field_desc
        self.biz_tag = biz_tag
        self.domain = domain
        self.domain_type = domain_type
        self.collection_id = collection_id
        self.data_key = data_key
        self.field_order = field_order
        self.visit_info = visit_info

    def validate(self):
        pass

    def to_map(self):
        _map = super().to_map()
        if _map is not None:
            return _map

        result = dict()
        if self.id is not None:
            result['id'] = self.id
        if self.gmt_create is not None:
            result['gmtCreate'] = self.gmt_create
        if self.gmt_modified is not None:
            result['gmtModified'] = self.gmt_modified
        if self.creator_id is not None:
            result['creatorId'] = self.creator_id
        if self.modifier_id is not None:
            result['modifierId'] = self.modifier_id
        if self.field_name is not None:
            result['fieldName'] = self.field_name
        if self.type is not None:
            result['type'] = self.type
        if self.field_type is not None:
            result['fieldType'] = self.field_type
        if self.classify is not None:
            result['classify'] = self.classify
        if self.field_classify is not None:
            result['fieldClassify'] = self.field_classify
        if self.category is not None:
            result['category'] = self.category
        if self.alias is not None:
            result['alias'] = self.alias
        if self.nullable is not None:
            result['nullable'] = self.nullable
        if self.field_sensitive is not None:
            result['fieldSensitive'] = self.field_sensitive
        if self.defval is not None:
            result['defval'] = self.defval
        if self.field_primary is not None:
            result['fieldPrimary'] = self.field_primary
        if self.field_desc is not None:
            result['fieldDesc'] = self.field_desc
        if self.biz_tag is not None:
            result['bizTag'] = self.biz_tag
        if self.domain is not None:
            result['domain'] = self.domain
        if self.domain_type is not None:
            result['domainType'] = self.domain_type
        if self.collection_id is not None:
            result['collectionId'] = self.collection_id
        if self.data_key is not None:
            result['dataKey'] = self.data_key
        if self.field_order is not None:
            result['fieldOrder'] = self.field_order
        if self.visit_info is not None:
            result['visitInfo'] = self.visit_info
        return result

    def from_map(self, m: dict = None):
        m = m or dict()
        if m.get('id') is not None:
            self.id = m.get('id')
        if m.get('gmtCreate') is not None:
            self.gmt_create = m.get('gmtCreate')
        if m.get('gmtModified') is not None:
            self.gmt_modified = m.get('gmtModified')
        if m.get('creatorId') is not None:
            self.creator_id = m.get('creatorId')
        if m.get('modifierId') is not None:
            self.modifier_id = m.get('modifierId')
        if m.get('fieldName') is not None:
            self.field_name = m.get('fieldName')
        if m.get('type') is not None:
            self.type = m.get('type')
        if m.get('fieldType') is not None:
            self.field_type = m.get('fieldType')
        if m.get('classify') is not None:
            self.classify = m.get('classify')
        if m.get('fieldClassify') is not None:
            self.field_classify = m.get('fieldClassify')
        if m.get('category') is not None:
            self.category = m.get('category')
        if m.get('alias') is not None:
            self.alias = m.get('alias')
        if m.get('nullable') is not None:
            self.nullable = m.get('nullable')
        if m.get('fieldSensitive') is not None:
            self.field_sensitive = m.get('fieldSensitive')
        if m.get('defval') is not None:
            self.defval = m.get('defval')
        if m.get('fieldPrimary') is not None:
            self.field_primary = m.get('fieldPrimary')
        if m.get('fieldDesc') is not None:
            self.field_desc = m.get('fieldDesc')
        if m.get('bizTag') is not None:
            self.biz_tag = m.get('bizTag')
        if m.get('domain') is not None:
            self.domain = m.get('domain')
        if m.get('domainType') is not None:
            self.domain_type = m.get('domainType')
        if m.get('collectionId') is not None:
            self.collection_id = m.get('collectionId')
        if m.get('dataKey') is not None:
            self.data_key = m.get('dataKey')
        if m.get('fieldOrder') is not None:
            self.field_order = m.get('fieldOrder')
        if m.get('visitInfo') is not None:
            self.visit_info = m.get('visitInfo')
        return self


class GetInitCollectionSchemaBody(TeaModel):
    def __init__(
        self,
        code: int = None,
        desc: str = None,
        err_msg: str = None,
        succ: bool = None,
        host: str = None,
        trace_id: str = None,
        time: int = None,
        result: List[FieldInfo] = None,
    ):
        self.code = code
        self.desc = desc
        self.err_msg = err_msg
        self.succ = succ
        self.host = host
        self.trace_id = trace_id
        self.time = time
        self.result = result

    def validate(self):
        if self.result:
            for k in self.result:
                if k:
                    k.validate()

    def to_map(self):
        _map = super().to_map()
        if _map is not None:
            return _map

        result = dict()
        if self.code is not None:
            result['code'] = self.code
        if self.desc is not None:
            result['desc'] = self.desc
        if self.err_msg is not None:
            result['errMsg'] = self.err_msg
        if self.succ is not None:
            result['succ'] = self.succ
        if self.host is not None:
            result['host'] = self.host
        if self.trace_id is not None:
            result['traceId'] = self.trace_id
        if self.time is not None:
            result['time'] = self.time
        result['result'] = []
        if self.result is not None:
            for k in self.result:
                result['result'].append(k.to_map() if k else None)
        return result

    def from_map(self, m: dict = None):
        m = m or dict()
        if m.get('code') is not None:
            self.code = m.get('code')
        if m.get('desc') is not None:
            self.desc = m.get('desc')
        if m.get('errMsg') is not None:
            self.err_msg = m.get('errMsg')
        if m.get('succ') is not None:
            self.succ = m.get('succ')
        if m.get('host') is not None:
            self.host = m.get('host')
        if m.get('traceId') is not None:
            self.trace_id = m.get('traceId')
        if m.get('time') is not None:
            self.time = m.get('time')
        self.result = []
        if m.get('result') is not None:
            for k in m.get('result'):
                temp_model = FieldInfo()
                self.result.append(temp_model.from_map(k))
        return self


class GetInitCollectionSchemaResponse(TeaModel):
    def __init__(
        self,
        headers: Dict[str, str] = None,
        status_code: int = None,
        body: GetInitCollectionSchemaBody = None,
    ):
        self.headers = headers
        self.status_code = status_code
        self.body = body

    def validate(self):
        self.validate_required(self.headers, 'headers')
        self.validate_required(self.status_code, 'status_code')
        self.validate_required(self.body, 'body')
        if self.body:
            self.body.validate()

    def to_map(self):
        _map = super().to_map()
        if _map is not None:
            return _map

        result = dict()
        if self.headers is not None:
            result['headers'] = self.headers
        if self.status_code is not None:
            result['statusCode'] = self.status_code
        if self.body is not None:
            result['body'] = self.body.to_map()
        return result

    def from_map(self, m: dict = None):
        m = m or dict()
        if m.get('headers') is not None:
            self.headers = m.get('headers')
        if m.get('statusCode') is not None:
            self.status_code = m.get('statusCode')
        if m.get('body') is not None:
            temp_model = GetInitCollectionSchemaBody()
            self.body = temp_model.from_map(m['body'])
        return self


class StsConf(TeaModel):
    def __init__(
        self,
        access_id: str = None,
        policy: str = None,
        signature: str = None,
        dir: str = None,
        host: str = None,
        expire: int = None,
        callback: str = None,
    ):
        self.access_id = access_id
        self.policy = policy
        self.signature = signature
        self.dir = dir
        self.host = host
        self.expire = expire
        self.callback = callback

    def validate(self):
        pass

    def to_map(self):
        _map = super().to_map()
        if _map is not None:
            return _map

        result = dict()
        if self.access_id is not None:
            result['accessId'] = self.access_id
        if self.policy is not None:
            result['policy'] = self.policy
        if self.signature is not None:
            result['signature'] = self.signature
        if self.dir is not None:
            result['dir'] = self.dir
        if self.host is not None:
            result['host'] = self.host
        if self.expire is not None:
            result['expire'] = self.expire
        if self.callback is not None:
            result['callback'] = self.callback
        return result

    def from_map(self, m: dict = None):
        m = m or dict()
        if m.get('accessId') is not None:
            self.access_id = m.get('accessId')
        if m.get('policy') is not None:
            self.policy = m.get('policy')
        if m.get('signature') is not None:
            self.signature = m.get('signature')
        if m.get('dir') is not None:
            self.dir = m.get('dir')
        if m.get('host') is not None:
            self.host = m.get('host')
        if m.get('expire') is not None:
            self.expire = m.get('expire')
        if m.get('callback') is not None:
            self.callback = m.get('callback')
        return self


class GetOssPostSignatureBody(TeaModel):
    def __init__(
        self,
        code: int = None,
        desc: str = None,
        err_msg: str = None,
        succ: bool = None,
        host: str = None,
        trace_id: str = None,
        time: int = None,
        result: StsConf = None,
    ):
        self.code = code
        self.desc = desc
        self.err_msg = err_msg
        self.succ = succ
        self.host = host
        self.trace_id = trace_id
        self.time = time
        self.result = result

    def validate(self):
        if self.result:
            self.result.validate()

    def to_map(self):
        _map = super().to_map()
        if _map is not None:
            return _map

        result = dict()
        if self.code is not None:
            result['code'] = self.code
        if self.desc is not None:
            result['desc'] = self.desc
        if self.err_msg is not None:
            result['errMsg'] = self.err_msg
        if self.succ is not None:
            result['succ'] = self.succ
        if self.host is not None:
            result['host'] = self.host
        if self.trace_id is not None:
            result['traceId'] = self.trace_id
        if self.time is not None:
            result['time'] = self.time
        if self.result is not None:
            result['result'] = self.result.to_map()
        return result

    def from_map(self, m: dict = None):
        m = m or dict()
        if m.get('code') is not None:
            self.code = m.get('code')
        if m.get('desc') is not None:
            self.desc = m.get('desc')
        if m.get('errMsg') is not None:
            self.err_msg = m.get('errMsg')
        if m.get('succ') is not None:
            self.succ = m.get('succ')
        if m.get('host') is not None:
            self.host = m.get('host')
        if m.get('traceId') is not None:
            self.trace_id = m.get('traceId')
        if m.get('time') is not None:
            self.time = m.get('time')
        if m.get('result') is not None:
            temp_model = StsConf()
            self.result = temp_model.from_map(m['result'])
        return self


class GetOssPostSignatureResponse(TeaModel):
    def __init__(
        self,
        headers: Dict[str, str] = None,
        status_code: int = None,
        body: GetOssPostSignatureBody = None,
    ):
        self.headers = headers
        self.status_code = status_code
        self.body = body

    def validate(self):
        self.validate_required(self.headers, 'headers')
        self.validate_required(self.status_code, 'status_code')
        self.validate_required(self.body, 'body')
        if self.body:
            self.body.validate()

    def to_map(self):
        _map = super().to_map()
        if _map is not None:
            return _map

        result = dict()
        if self.headers is not None:
            result['headers'] = self.headers
        if self.status_code is not None:
            result['statusCode'] = self.status_code
        if self.body is not None:
            result['body'] = self.body.to_map()
        return result

    def from_map(self, m: dict = None):
        m = m or dict()
        if m.get('headers') is not None:
            self.headers = m.get('headers')
        if m.get('statusCode') is not None:
            self.status_code = m.get('statusCode')
        if m.get('body') is not None:
            temp_model = GetOssPostSignatureBody()
            self.body = temp_model.from_map(m['body'])
        return self


class CreateSdkLogResponse(TeaModel):
    def __init__(
        self,
        status_code: int = None,
    ):
        self.status_code = status_code

    def validate(self):
        pass

    def to_map(self):
        _map = super().to_map()
        if _map is not None:
            return _map

        result = dict()
        if self.status_code is not None:
            result['statusCode'] = self.status_code
        return result

    def from_map(self, m: dict = None):
        m = m or dict()
        if m.get('statusCode') is not None:
            self.status_code = m.get('statusCode')
        return self


class UploadFileFromStsConfResponse(TeaModel):
    def __init__(
        self,
        status_code: int = None,
    ):
        self.status_code = status_code

    def validate(self):
        pass

    def to_map(self):
        _map = super().to_map()
        if _map is not None:
            return _map

        result = dict()
        if self.status_code is not None:
            result['statusCode'] = self.status_code
        return result

    def from_map(self, m: dict = None):
        m = m or dict()
        if m.get('statusCode') is not None:
            self.status_code = m.get('statusCode')
        return self


class CreateDatasetRequest(TeaModel):
    def __init__(
        self,
        data_source: str = None,
        dataset_name: str = None,
        owner_name: str = None,
        owner_employee_id: str = None,
        oss_conf: OssConf = None,
        file_name: str = None,
        file: BinaryIO = None,
        content_type: str = None,
        secure_level: int = None,
        remark: str = None,
    ):
        self.data_source = data_source
        self.dataset_name = dataset_name
        self.owner_name = owner_name
        self.owner_employee_id = owner_employee_id
        self.oss_conf = oss_conf
        self.file_name = file_name
        self.file = file
        self.content_type = content_type
        self.secure_level = secure_level
        self.remark = remark

    def validate(self):
        self.validate_required(self.data_source, 'data_source')
        self.validate_required(self.dataset_name, 'dataset_name')
        self.validate_required(self.owner_name, 'owner_name')
        self.validate_required(self.owner_employee_id, 'owner_employee_id')
        if self.oss_conf:
            self.oss_conf.validate()
        self.validate_required(self.secure_level, 'secure_level')
        self.validate_required(self.remark, 'remark')

    def to_map(self):
        _map = super().to_map()
        if _map is not None:
            return _map

        result = dict()
        if self.data_source is not None:
            result['dataSource'] = self.data_source
        if self.dataset_name is not None:
            result['datasetName'] = self.dataset_name
        if self.owner_name is not None:
            result['ownerName'] = self.owner_name
        if self.owner_employee_id is not None:
            result['ownerEmployeeId'] = self.owner_employee_id
        if self.oss_conf is not None:
            result['ossConf'] = self.oss_conf.to_map()
        if self.file_name is not None:
            result['fileName'] = self.file_name
        if self.file is not None:
            result['file'] = self.file
        if self.content_type is not None:
            result['contentType'] = self.content_type
        if self.secure_level is not None:
            result['secureLevel'] = self.secure_level
        if self.remark is not None:
            result['remark'] = self.remark
        return result

    def from_map(self, m: dict = None):
        m = m or dict()
        if m.get('dataSource') is not None:
            self.data_source = m.get('dataSource')
        if m.get('datasetName') is not None:
            self.dataset_name = m.get('datasetName')
        if m.get('ownerName') is not None:
            self.owner_name = m.get('ownerName')
        if m.get('ownerEmployeeId') is not None:
            self.owner_employee_id = m.get('ownerEmployeeId')
        if m.get('ossConf') is not None:
            temp_model = OssConf()
            self.oss_conf = temp_model.from_map(m['ossConf'])
        if m.get('fileName') is not None:
            self.file_name = m.get('fileName')
        if m.get('file') is not None:
            self.file = m.get('file')
        if m.get('contentType') is not None:
            self.content_type = m.get('contentType')
        if m.get('secureLevel') is not None:
            self.secure_level = m.get('secureLevel')
        if m.get('remark') is not None:
            self.remark = m.get('remark')
        return self


class CreateDatasetBody(TeaModel):
    def __init__(
        self,
        code: int = None,
        desc: str = None,
        err_msg: str = None,
        succ: bool = None,
        host: str = None,
        trace_id: str = None,
        time: int = None,
        result: int = None,
    ):
        self.code = code
        self.desc = desc
        self.err_msg = err_msg
        self.succ = succ
        self.host = host
        self.trace_id = trace_id
        self.time = time
        self.result = result

    def validate(self):
        pass

    def to_map(self):
        _map = super().to_map()
        if _map is not None:
            return _map

        result = dict()
        if self.code is not None:
            result['code'] = self.code
        if self.desc is not None:
            result['desc'] = self.desc
        if self.err_msg is not None:
            result['errMsg'] = self.err_msg
        if self.succ is not None:
            result['succ'] = self.succ
        if self.host is not None:
            result['host'] = self.host
        if self.trace_id is not None:
            result['traceId'] = self.trace_id
        if self.time is not None:
            result['time'] = self.time
        if self.result is not None:
            result['result'] = self.result
        return result

    def from_map(self, m: dict = None):
        m = m or dict()
        if m.get('code') is not None:
            self.code = m.get('code')
        if m.get('desc') is not None:
            self.desc = m.get('desc')
        if m.get('errMsg') is not None:
            self.err_msg = m.get('errMsg')
        if m.get('succ') is not None:
            self.succ = m.get('succ')
        if m.get('host') is not None:
            self.host = m.get('host')
        if m.get('traceId') is not None:
            self.trace_id = m.get('traceId')
        if m.get('time') is not None:
            self.time = m.get('time')
        if m.get('result') is not None:
            self.result = m.get('result')
        return self


class CreateDatasetResponse(TeaModel):
    def __init__(
        self,
        headers: Dict[str, str] = None,
        status_code: int = None,
        body: CreateDatasetBody = None,
    ):
        self.headers = headers
        self.status_code = status_code
        self.body = body

    def validate(self):
        self.validate_required(self.headers, 'headers')
        self.validate_required(self.status_code, 'status_code')
        self.validate_required(self.body, 'body')
        if self.body:
            self.body.validate()

    def to_map(self):
        _map = super().to_map()
        if _map is not None:
            return _map

        result = dict()
        if self.headers is not None:
            result['headers'] = self.headers
        if self.status_code is not None:
            result['statusCode'] = self.status_code
        if self.body is not None:
            result['body'] = self.body.to_map()
        return result

    def from_map(self, m: dict = None):
        m = m or dict()
        if m.get('headers') is not None:
            self.headers = m.get('headers')
        if m.get('statusCode') is not None:
            self.status_code = m.get('statusCode')
        if m.get('body') is not None:
            temp_model = CreateDatasetBody()
            self.body = temp_model.from_map(m['body'])
        return self


