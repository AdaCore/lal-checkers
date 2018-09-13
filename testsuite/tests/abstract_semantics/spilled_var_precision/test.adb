procedure Test is
   X : aliased Integer;
   Y : access all Integer := X'Access;
begin
   if X = 2 then
      Y := null;
   end if;
end Ex1;
