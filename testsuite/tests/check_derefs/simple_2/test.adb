procedure Test is
   type Int_Access is access Integer;
   X : Int_Access;
begin
   if X = null then
      X.all := 1;  -- null dereference
   end if;
end Test;