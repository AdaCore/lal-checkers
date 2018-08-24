procedure Test is
   procedure P is
   begin
      null;
   end;

   F : access procedure := null;
   B : Boolean;
begin
   if B then
      F := P'Access;
   end if;

   F.all;
end Ex1;
